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
import re
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
QUICK_MAX_TOKENS = 1024    # an answer this short has no room to think first (agents' side queries: a forced tool call
#                            in 256 tokens - with thinking on, a long conversation used them all and no call came)
REASONING_KEYS = ("reasoning_effort", "reasoning", "enable_thinking", "chat_template_kwargs", "thinking")
FIT_RE = re.compile(r"exceeds the context \(\d+\).*?at most (\d+) here")   # serve/server.py's prepare() refusal
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

    @staticmethod
    def count_videos(obj) -> int:
        if isinstance(obj, list):
            return sum(Encoder.count_videos(x) for x in obj)
        if not isinstance(obj, dict):
            return 0
        if obj.get("type") in VIDEO_PARTS:
            return 1
        return sum(Encoder.count_videos(v) for k, v in obj.items() if k in ("messages", "input", "content"))

    def rewrite(self, obj, top: bool = True):
        """Every content list in the request (messages, Responses input items, tool results inside them).  The
        request's videos share the automatic video budget (two videos compared: half each)."""
        if top:
            self.vision.videos_in_request = max(1, self.count_videos(obj))
        if isinstance(obj, list):
            return [self.rewrite(x, False) for x in obj]
        if not isinstance(obj, dict):
            return obj
        if obj.get("type") in IMAGE_PARTS + VIDEO_PARTS:
            return self.part(obj)
        return {k: (self.rewrite(v, False) if k in ("messages", "input", "content") else v) for k, v in obj.items()}


def quick(req: dict) -> int:
    """A chat request that allows at most QUICK_MAX_TOKENS and says nothing about thinking -> its limit, else 0."""
    limit = next((req[k] for k in ("max_tokens", "max_completion_tokens") if isinstance(req.get(k), int)), None)
    if limit is None or limit > QUICK_MAX_TOKENS or any(k in req for k in REASONING_KEYS):
        return 0
    return limit


def make_handler(upstream: Upstream, encoder: Encoder, own_origins: set[str], log=None):
    def note(msg: str):
        if log is not None:
            log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            log.flush()

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                  # the answer ends when the connection closes: streams pass

        def log_message(self, fmt, *args):            # quiet (--log writes each request); errors reach the client
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

        def _fit(self, body: bytes, err: bytes) -> bytes | None:
            """A request refused only because prompt + max tokens passes the context: the same request with the
            max tokens the server says still fit (it never truncates; agent clients ask for 32K by habit)."""
            m = FIT_RE.search(err.decode("utf-8", "replace"))
            if not m or int(m.group(1)) < 256:
                return None
            try:
                req = json.loads(body)
            except ValueError:
                return None
            keys = [k for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if k in req] or ["max_tokens"]
            for k in keys:
                req[k] = int(m.group(1))
            return json.dumps(req).encode()

        def _forward(self, body: bytes | None, retry: bool = True):
            c = upstream.conn()
            t0, sent, status, started = time.monotonic(), 0, None, False
            try:
                c.request(self.command, self.path, body=body, headers=self._headers())
                r = c.getresponse()
                status = r.status
                if status == 400 and retry and body and self.command == "POST":
                    err = r.read()
                    fitted = self._fit(body, err)
                    if fitted is not None:
                        note(f"{self.command} {self.path}: answer length shortened to fit the context, sent again")
                        c.close()
                        return self._forward(fitted, retry=False)
                    self.send_response(r.status, r.reason)
                    self.send_header("Content-Type", r.getheader("Content-Type") or "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    note(f"{self.command} {self.path} -> 400: {err[:2048].decode('utf-8', 'replace')}")
                    return
                self.send_response(r.status, r.reason)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                if r.getheader("Content-Length"):
                    self.send_header("Content-Length", r.getheader("Content-Length"))
                self.end_headers()
                started = True
                head = b""
                while True:                            # piece by piece: a streamed answer arrives as it is made
                    chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    sent += len(chunk)
                    if status >= 400 and len(head) < 2048:
                        head += chunk[:2048 - len(head)]
                note(f"{self.command} {self.path} -> {status}, {sent:,} bytes in {time.monotonic() - t0:.1f} s"
                     + (f": {head.decode('utf-8', 'replace')}" if head else ""))
            except OSError as e:
                note(f"{self.command} {self.path} -> {status}, {sent:,} bytes, broken after {time.monotonic() - t0:.1f} s:"
                     f" {type(e).__name__}: {e}")
                if not started:
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
                t0 = time.monotonic()
                try:
                    req = json.loads(body)
                    if isinstance(req, dict):
                        req = encoder.rewrite(copy.deepcopy(req))
                        if path == "/v1/chat/completions" and quick(req):
                            req["reasoning_effort"] = "none"
                            note(f"POST {path}: a short answer ({quick(req)} tokens): thinking off")
                        body = json.dumps(req).encode()
                except json.JSONDecodeError:
                    pass                                # the server says what is wrong with it
                except ValueError as e:
                    note(f"POST {path}: refused while encoding: {e}")
                    self._error(400, str(e))
                    return
                except Exception as e:                  # anything else: say so, never drop the connection silently
                    import traceback
                    note(f"POST {path}: failed while encoding:\n{traceback.format_exc()}")
                    self._error(500, f"the video proxy failed while encoding this request: {type(e).__name__}: {e}")
                    return
                note(f"POST {path}: {len(body):,} bytes after encoding ({time.monotonic() - t0:.1f} s)")
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
    ap.add_argument("--log", help="write each request (size, status, time, errors) to this file")
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
    log = open(a.log, "a", encoding="utf-8") if a.log else None
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(upstream, encoder, own, log))
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
