"""serve/embeddings.py - images and videos encoded on another PC (strata-video).

A PC with the vision encoder (tools/video_proxy.py) sends what `strata-vision` made of a picture or a video as a
*bundle*; the server keeps it and a request names it by id in an `image_embeddings` / `video_embeddings` part.  The
server needs no encoder, no ffmpeg and no mmproj for that ("vision": {"remote": true} in the config).

A bundle is  b"SVB1", uint32 header length, a JSON header, then the SVE1 records strata-vision writes
(int32 {0x31455653, n, nx, ny, n_embd} + rows), their rows in the header's dtype ("f16" halves the upload; "f32" is
bit-exact).  The header: {"v": 1, "kind": "image" | "video", "n": cells, "n_embd": width, "dtype": ...,
"layout": the video's layout (lists of text token ids and image cell counts, as Vision.parse_layout makes it) or
null, "mmproj_sha256": the encoder file's hash or null}.  Its id is the SHA-256 of the whole bundle.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from pathlib import Path

MAGIC = b"SVB1"
RECORD = 0x31455653                     # 'SVE1', strata-vision's record header
REF = "strata-embeddings:"              # the source string an embeddings part becomes (frontend -> Vision)
DTYPES = {"f16": "<f2", "f32": "<f4"}


def make_bundle(kind: str, sve: bytes, n: int, layout=None, dtype: str = "f16", mmproj_sha256: str | None = None) -> bytes:
    """strata-vision's output file (f32 SVE1 records) -> a bundle; f16 rows halve it."""
    import numpy as np
    out, off, n_embd = [], 0, 0
    while off < len(sve):
        magic, rn, nx, ny, n_embd = struct.unpack_from("<5i", sve, off)
        if magic != RECORD:
            raise ValueError("not a strata-vision embeddings file")
        rows = np.frombuffer(sve, "<f4", rn * n_embd, off + 20)
        out.append(struct.pack("<5i", magic, rn, nx, ny, n_embd) + rows.astype(DTYPES[dtype]).tobytes())
        off += 20 + rn * n_embd * 4
    header = json.dumps({"v": 1, "kind": kind, "n": n, "n_embd": n_embd, "dtype": dtype, "layout": layout,
                         "mmproj_sha256": mmproj_sha256}, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<I", len(header)) + header + b"".join(out)


def parse_bundle(data: bytes):
    """-> (header, SVE1 file bytes in f32, as the engine reads them).  ValueError on anything malformed."""
    import numpy as np
    if data[:4] != MAGIC or len(data) < 8:
        raise ValueError("not a Strata embeddings bundle")
    (hl,) = struct.unpack_from("<I", data, 4)
    try:
        h = json.loads(data[8:8 + hl])
    except ValueError:
        raise ValueError("the bundle's header is not JSON") from None
    if not isinstance(h, dict) or h.get("v") != 1 or h.get("kind") not in ("image", "video") or \
            h.get("dtype") not in DTYPES:
        raise ValueError("unsupported bundle (version, kind or dtype)")
    n, n_embd = h.get("n"), h.get("n_embd")
    if not (isinstance(n, int) and n > 0 and isinstance(n_embd, int) and n_embd > 0):
        raise ValueError("the bundle's n / n_embd are invalid")
    item = np.dtype(DTYPES[h["dtype"]]).itemsize
    off, out, cells, records = 8 + hl, [], [], 0
    while off < len(data):
        if off + 20 > len(data):
            raise ValueError("the bundle ends inside a record header")
        magic, rn, nx, ny, w = struct.unpack_from("<5i", data, off)
        if magic != RECORD or rn < 1 or nx < 1 or ny < 1 or nx * ny != rn or w != n_embd:
            raise ValueError("a record of the bundle is malformed")
        size = rn * w * item
        if off + 20 + size > len(data):
            raise ValueError("the bundle is shorter than its records")
        rows = np.frombuffer(data, DTYPES[h["dtype"]], rn * w, off + 20)
        out.append(struct.pack("<5i", magic, rn, nx, ny, w) + rows.astype("<f4").tobytes())
        cells.append(rn)
        records += 1
        off += 20 + size
    if sum(cells) != n:
        raise ValueError("the bundle's records do not add up to its n")
    lay = h.get("layout")
    if h["kind"] == "image":
        if records != 1 or lay is not None:
            raise ValueError("an image bundle holds one record and no layout")
    else:
        if not isinstance(lay, list) or not all(isinstance(x, int) or
                                                (isinstance(x, list) and all(isinstance(t, int) for t in x))
                                                for x in lay):
            raise ValueError("a video bundle needs its layout")
        if [x for x in lay if isinstance(x, int)] != cells:
            raise ValueError("the video's layout does not match its records")
    return h, b"".join(out)


class EmbeddingStore:
    """Bundles received, as engine-ready files: <id>.sve (f32 SVE1) + <id>.json (the header).  Kept on disk up to
    max_bytes; the least recently used go first."""

    def __init__(self, directory: Path, max_bytes: int = 8 << 30, n_embd: int | None = None,
                 mmproj_sha256: str | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.n_embd, self.mmproj = max_bytes, n_embd, mmproj_sha256
        self.lock = threading.Lock()

    def _paths(self, bid: str):
        if len(bid) != 64 or any(c not in "0123456789abcdef" for c in bid):
            raise ValueError("an embeddings id is 64 hex characters")
        return self.dir / f"{bid}.sve", self.dir / f"{bid}.json"

    def has(self, bid: str) -> bool:
        sve, meta = self._paths(bid)
        return sve.exists() and meta.exists()

    def put(self, data: bytes) -> dict:
        """Check and keep a bundle -> {"id", "kind", "n"}."""
        bid = hashlib.sha256(data).hexdigest()
        h, sve = parse_bundle(data)
        if self.n_embd and h["n_embd"] != self.n_embd:
            raise ValueError(f"these embeddings are {h['n_embd']} wide; this model reads {self.n_embd}")
        if self.mmproj and h.get("mmproj_sha256") and h["mmproj_sha256"] != self.mmproj:
            raise ValueError("these embeddings come from another vision encoder (mmproj) than this model's")
        p_sve, p_meta = self._paths(bid)
        with self.lock:
            p_sve.write_bytes(sve)
            p_meta.write_text(json.dumps({k: h[k] for k in ("kind", "n", "layout", "n_embd", "dtype")}))
            self._trim(keep=bid)
        return {"id": bid, "kind": h["kind"], "n": h["n"]}

    def get(self, bid: str, kind: str):
        """-> (SVE1 path, cells, layout or None).  ValueError when unknown or of the other kind."""
        p_sve, p_meta = self._paths(bid)
        if not (p_sve.exists() and p_meta.exists()):
            raise ValueError(f"no embeddings {bid[:12]}... on this server (upload them first, or send the file again)")
        m = json.loads(p_meta.read_text())
        if m["kind"] != kind:
            raise ValueError(f"embeddings {bid[:12]}... are a {m['kind']}, not a {kind}")
        os.utime(p_sve)                                 # recently used
        return p_sve, int(m["n"]), m.get("layout")

    def _trim(self, keep: str):
        files = sorted(self.dir.glob("*.sve"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for p in files:
            if total <= self.max_bytes or p.stem == keep:
                continue
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)


def lookup(store: EmbeddingStore | None, source: str, kind: str):
    """A part's source that names stored embeddings -> what Vision.encode / encode_video return; None otherwise."""
    if not source.startswith(REF):
        return None
    if store is None:
        raise ValueError("this server takes no pre-encoded images or videos")
    path, n, layout = store.get(source[len(REF):], kind)
    return (path, n) if kind == "image" else (path, n, layout)


class RemoteVision:
    """"vision": {"remote": true}: no encoder here - pictures and videos arrive encoded (tools/video_proxy.py on a PC
    with the encoder).  Quacks like Vision for the server; anything not pre-encoded is refused with how to send it."""

    def __init__(self, store: EmbeddingStore, directory: Path):
        self.dir = Path(directory)                     # the server writes each request's combined file here
        self.store = store
        self.lock = threading.Lock()

    @staticmethod
    def _refuse(what):
        raise ValueError(f"this server encodes no {what} itself (vision remote): send them through Strata's video "
                         "proxy on a PC with the encoder (tools/video_proxy.py), which uploads them encoded")

    def encode(self, source: str):
        return lookup(self.store, source, "image") or self._refuse("pictures")

    def encode_video(self, source: str, opts: dict | None = None):
        return lookup(self.store, source, "video") or self._refuse("videos")

    def alive(self) -> bool:
        return True

    def unload(self):
        pass

    def restart(self):
        pass

    def close(self):
        pass
