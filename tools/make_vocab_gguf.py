"""tools/make_vocab_gguf.py - the model's vocabulary as a small GGUF, for a PC that only encodes (video_proxy.py).

strata-vision opens the text model vocabulary-only (no weights), so a PC that encodes pictures and videos for a
server does not need the 60-110 GB model: this copies the first shard's metadata (tokenizer included, ~11 MB) into a
GGUF without tensors.  The source is a local first shard or its download URL (only the first 64 MB are fetched).

    python tools/make_vocab_gguf.py https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf vocab-only.gguf
"""
from __future__ import annotations

import struct
import sys
import urllib.request

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # GGUF scalar value types


def head(src: str, size: int = 64 << 20) -> bytes:
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"Range": f"bytes=0-{size - 1}", "User-Agent": "strata"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read(size)
    with open(src, "rb") as f:
        return f.read(size)


def vocab_gguf(buf: bytes) -> tuple[bytes, int]:
    """-> (a tensor-less GGUF with every key but split.*, number of keys)."""
    pos = 0

    def take(n):
        nonlocal pos
        if pos + n > len(buf):
            raise SystemExit("the metadata is longer than what was read")
        pos += n
        return buf[pos - n:pos]

    def u32():
        return struct.unpack("<I", take(4))[0]

    def u64():
        return struct.unpack("<Q", take(8))[0]

    def skip(t):
        if t in SIZES:
            take(SIZES[t])
        elif t == 8:
            take(u64())
        elif t == 9:
            et, n = u32(), u64()
            for _ in range(n):
                skip(et)
        else:
            raise SystemExit(f"unknown GGUF value type {t}")

    if take(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    version = u32()
    u64()                                                # tensors: dropped
    n_kv = u64()
    kept = []
    for _ in range(n_kv):
        start = pos
        key = take(u64())
        skip(u32())
        if not key.startswith(b"split."):
            kept.append(buf[start:pos])
    return b"GGUF" + struct.pack("<IQQ", version, 0, len(kept)) + b"".join(kept), len(kept)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        raise SystemExit("usage: make_vocab_gguf.py <first shard path or URL> <out.gguf>")
    data, n = vocab_gguf(head(argv[0]))
    with open(argv[1], "wb") as f:
        f.write(data)
    print(f"{argv[1]}: {n} keys, {len(data) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
