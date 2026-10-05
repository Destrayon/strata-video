"""tools/video_proxy.py - encode pictures and videos on this PC, run the model on a Strata server elsewhere.

Runs this PC's `strata-vision` (the model's vision encoder) and serves Strata's API and web page on a local port.
Every picture and video in a request (OpenAI image_url / video_url, Responses input_image / input_video, Anthropic
image / video blocks) is encoded here, uploaded to the server once (POST /v1/strata/embeddings) and replaced by an
image_embeddings / video_embeddings part; everything else - the web page, /health, streaming answers - passes
through.  The server needs no encoder ("vision": {"remote": true}, setup --vision remote).

    python tools/video_proxy.py --server http://my-server:8080 --config strata-iq2_xs.json
    python tools/video_proxy.py --server http://my-server:8080 --exe engine/strata-vision.exe --gpu \
        --mmproj mmproj-Qwen3.8-Flash-Next-BF16.gguf --model vocab-only.gguf

Then point the browser or apps at http://127.0.0.1:8090 instead of the server.  --config takes the "vision" section
of a Strata run config on this PC (encoder, mmproj, model, GPU, token cap, video defaults); the model may be a
tensor-less GGUF (tools/make_vocab_gguf.py), since the encoder reads only its vocabulary.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import embeddings as E  # noqa: E402
from serve.frontend import IMAGE_PARTS, VIDEO_PARTS, _image_source, _video_source  # noqa: E402

ENCODED = ("image_embeddings", "video_embeddings")
REWRITE = ("/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses")
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "proxy-connection", "upgrade",
       "accept-encoding"}


class Upstream:
    """The Strata server: uploads (once per bundle) and plain requests."""

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 3600):
        u = urlsplit(url.rstrip("/"))
        if u.scheme not in ("http", "https") or not u.hostname:
            raise SystemExit(f"--server must be an http(s) URL, not {url!r}")
        self.scheme, self.host, self.port = u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.origin = f"{u.scheme}://{u.netloc}"
        self.api_key, self.timeout = api_key, timeout

    def conn(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def auth(self, headers: dict) -> dict:
        if self.api_key and not any(k.lower() in ("authorization", "x-api-key") for k in headers):
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def context(self) -> int:
        """The server's context in tokens (its /health), 0 when unknown; asked at most once a minute."""
        now = time.monotonic()
        if now - getattr(self, "_ctx_at", -1e9) > 60:
            c = self.conn()
            try:
                c.request("GET", "/health", headers=self.auth({}))
                r = c.getresponse()
                self._ctx = int(json.loads(r.read() or b"{}").get("max_context") or 0) if r.status == 200 else 0
            except (OSError, ValueError):
                self._ctx = 0
            finally:
                c.close()
            self._ctx_at = now
        return self._ctx

    def has(self, bid: str) -> bool:
        c = self.conn()
        try:
            c.request("GET", f"/v1/strata/embeddings/{bid}", headers=self.auth({}))
            r = c.getresponse()
            r.read()
            return r.status == 200
        finally:
            c.close()

    def upload(self, bundle: bytes) -> str:
        bid = hashlib.sha256(bundle).hexdigest()
        if self.has(bid):
            return bid
        c = self.conn()
        try:
            c.request("POST", "/v1/strata/embeddings", body=bundle,
                      headers=self.auth({"Content-Type": "application/octet-stream"}))
            r = c.getresponse()
            body = json.loads(r.read() or b"{}")
        finally:
            c.close()
        if r.status != 200:
            msg = (body.get("error") or {}).get("message") or f"HTTP {r.status}"
            raise ValueError(f"the server refused the encoded upload: {msg}")
        return body["id"]


class Encoder:
    """This PC's vision encoder (serve.server.Vision) plus the bundles made from its output, uploaded once."""

    def __init__(self, vision, upstream: Upstream, dtype: str = "f16", mmproj_sha256: str | None = None):
        self.vision, self.upstream, self.dtype, self.mmproj = vision, upstream, dtype, mmproj_sha256
        self.sent: dict[tuple, str] = {}               # (kind, encoder output file) -> id on the server
        self.lock = threading.Lock()

    def _send(self, kind, path, n, layout=None) -> str:
        key = (kind, str(path), self.upstream.origin)
        with self.lock:
            bid = self.sent.get(key)
        if bid and self.upstream.has(bid):             # sent before, and the server still has it
            return bid
        bid = self.upstream.upload(E.make_bundle(kind, Path(path).read_bytes(), n, layout, self.dtype, self.mmproj))
        with self.lock:
            self.sent[key] = bid
        return bid

    def part(self, part: dict) -> dict:
        """A picture or video part -> an embeddings part (the id of its upload); other parts unchanged."""
        t = part.get("type")
        if t in ENCODED or (t not in IMAGE_PARTS and t not in VIDEO_PARTS):
            return part
        if t in IMAGE_PARTS:
            path, n = self.vision.encode(_image_source(part))
            return {"type": "image_embeddings", "id": self._send("image", path, n)}
        src, opts = _video_source(part)
        ctx = self.upstream.context()
        if ctx:
            self.vision.context = ctx                  # the automatic video budget follows the server's context
        path, n, layout = self.vision.encode_video(src, opts)
        return {"type": "video_embeddings", "id": self._send("video", path, n, layout)}

    def rewrite(self, obj):
        """Every content list in the request (messages, Responses input items, tool results inside them)."""
        if isinstance(obj, list):
            return [self.rewrite(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        if obj.get("type") in IMAGE_PARTS + VIDEO_PARTS:
            return self.part(obj)
        return {k: (self.rewrite(v) if k in ("messages", "input", "content") else v) for k, v in obj.items()}


def make_handler(upstream: Upstream, encoder: Encoder, own_origins: set[str]):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                  # the answer ends when the connection closes: streams pass

        def log_message(self, fmt, *args):            # quiet; errors still reach the client
            pass

        def _headers(self) -> dict:
            h = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            # the web page served from here is the server's own page; any other site's Origin stays as it is, so
            # the server still refuses foreign pages
            for k in ("Origin", "Referer"):
                v = h.get(k) or ""
                for o in own_origins:
                    if v == o or v.startswith(o + "/"):
                        h[k] = upstream.origin + v[len(o):]
                        break
            h["Host"] = upstream.origin.split("://", 1)[1]
            return upstream.auth(h)

        def _error(self, code, msg):
            body = json.dumps({"error": {"type": "invalid_request_error", "message": msg}}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _forward(self, body: bytes | None):
            c = upstream.conn()
            try:
                c.request(self.command, self.path, body=body, headers=self._headers())
                r = c.getresponse()
                self.send_response(r.status, r.reason)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                if r.getheader("Content-Length"):
                    self.send_header("Content-Length", r.getheader("Content-Length"))
                self.end_headers()
                while True:                            # piece by piece: a streamed answer arrives as it is made
                    chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except OSError as e:
                try:
                    self._error(502, f"the Strata server at {upstream.origin} did not answer ({e})")
                except OSError:
                    pass
            finally:
                c.close()

        def _body(self) -> bytes | None:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n > 0 else None

        def do_GET(self):
            self._forward(None)

        def do_OPTIONS(self):
            self._forward(None)

        def do_POST(self):
            body = self._body()
            path = self.path.split("?")[0].rstrip("/")
            if path in REWRITE and body:
                try:
                    req = json.loads(body)
                    if isinstance(req, dict):
                        body = json.dumps(encoder.rewrite(copy.deepcopy(req))).encode()
                except json.JSONDecodeError:
                    pass                                # the server says what is wrong with it
                except ValueError as e:
                    self._error(400, str(e))
                    return
            self._forward(body)

    return Proxy


def mmproj_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", required=True, help="the Strata server, e.g. http://192.168.1.20:8080")
    ap.add_argument("--api-key", help="the server's API key (when it has one)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1", help="where the proxy listens (default: this PC only)")
    ap.add_argument("--config", help="a Strata run config on this PC; its \"vision\" section is used")
    ap.add_argument("--exe", help="strata-vision(.exe)")
    ap.add_argument("--mmproj", help="the model's mmproj GGUF")
    ap.add_argument("--model", help="the model's GGUF (first shard), or a tensor-less one from make_vocab_gguf.py")
    ap.add_argument("--gpu", action="store_true", help="run the encoder on the GPU")
    ap.add_argument("--max-tokens", type=int, help="the most tokens a picture or frame pair becomes")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16", help="upload precision (f16: half the size)")
    a = ap.parse_args(argv)

    vcfg = {}
    if a.config:
        vcfg = dict(json.loads(Path(a.config).read_text(encoding="utf-8")).get("vision") or {})
        if vcfg.get("remote"):
            raise SystemExit(f"{a.config} is a remote-vision config: the proxy needs one with this PC's encoder")
    for k in ("exe", "mmproj", "model", "max_tokens"):
        if getattr(a, k):
            vcfg[k] = getattr(a, k)
    if a.gpu:
        vcfg["gpu"] = True
    missing = [k for k in ("exe", "mmproj", "model") if not vcfg.get(k)]
    if missing:
        raise SystemExit("the encoder needs " + ", ".join("--" + k for k in missing) + " (or --config)")

    from serve.server import Vision                    # the same encoder process the server runs
    upstream = Upstream(a.server, a.api_key)
    print(f"hashing {vcfg['mmproj']} (the server checks that both sides use the same encoder) ...", flush=True)
    digest = mmproj_hash(vcfg["mmproj"])
    print("starting the vision encoder ...", flush=True)
    encoder = Encoder(Vision(vcfg), upstream, a.dtype, digest)
    own = {f"http://{h}:{a.port}" for h in ("127.0.0.1", "localhost", a.host)}
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(upstream, encoder, own))
    print(f"ready: http://{a.host}:{a.port} -> {upstream.origin} (pictures and videos encoded here, {a.dtype})",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        encoder.vision.close()


if __name__ == "__main__":
    main()
