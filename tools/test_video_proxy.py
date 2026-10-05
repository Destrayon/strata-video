"""tools/test_video_proxy.py - the encode-here proxy against a fake Strata server and a fake encoder (no GPU, no
network): which parts are encoded, that each upload happens once, pass-through, streaming and the page's Origin.

    python -m unittest tools.test_video_proxy -v
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import video_proxy as VP  # noqa: E402
from serve import embeddings as E  # noqa: E402


class FakeServer(BaseHTTPRequestHandler):
    """Records what the proxy sends; keeps uploads; streams an SSE answer for chat."""
    log, uploads = [], {}

    def log_message(self, *a):
        pass

    def _send(self, code, obj=None, raw=None, ctype="application/json"):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeServer.log.append(("GET", self.path, dict(self.headers), None))
        if self.path.startswith("/v1/strata/embeddings/"):
            bid = self.path.rsplit("/", 1)[1]
            self._send(200 if bid in FakeServer.uploads else 404, {"exists": bid in FakeServer.uploads})
        elif self.path == "/health":
            self._send(200, {"status": "ok", "max_context": 262144, "videos": True})
        else:
            self._send(200, raw=b"<html>strata</html>", ctype="text/html")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        FakeServer.log.append(("POST", self.path, dict(self.headers), body))
        if self.path == "/v1/strata/embeddings":
            h, _ = E.parse_bundle(body)
            import hashlib
            bid = hashlib.sha256(body).hexdigest()
            FakeServer.uploads[bid] = h
            self._send(200, {"id": bid, "kind": h["kind"], "n": h["n"]})
            return
        req = json.loads(body or b"{}")
        if req.get("max_tokens", 0) > 5000:                            # Strata's refusal, word for word
            self._send(400, {"error": {"type": "invalid_request_error", "message":
                             f"prompt (60000 tokens) + max tokens ({req['max_tokens']}) exceeds the context (65536); "
                             "requests are never truncated. Send a smaller max_tokens (at most 4512 here), or add "
                             "\"fit_max_tokens\": true to the model's strata-<model>.json to shorten it"}})
            return
        self.send_response(200)                                        # a streamed answer, no Content-Length
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(3):
            self.wfile.write(f"data: {{\"i\": {i}}}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


class FakeVision:
    def __init__(self, d):
        rec = lambda nx, ny: struct.pack("<5i", E.RECORD, nx * ny, nx, ny, 4) + b"\0" * (nx * ny * 4 * 4)  # noqa
        self.img, self.vid = Path(d) / "i.sve", Path(d) / "v.sve"
        self.img.write_bytes(rec(2, 1))
        self.vid.write_bytes(rec(2, 1) + rec(2, 2))
        self.calls = []

    def encode(self, source):
        self.calls.append(("image", source))
        return self.img, 2

    def encode_video(self, source, opts):
        self.calls.append(("video", source, opts))
        return self.vid, 6, [[1, 2], 2, [3], 4, [4]]


class Proxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        cls.d = tempfile.TemporaryDirectory()
        cls.vision = FakeVision(cls.d.name)
        upstream = VP.Upstream(f"http://127.0.0.1:{cls.up.server_address[1]}")
        cls.encoder = VP.Encoder(cls.vision, upstream, "f16", "enc")
        cls.px = ThreadingHTTPServer(("127.0.0.1", 0), VP.make_handler(upstream, cls.encoder, set()))
        port = cls.px.server_address[1]
        cls.px.RequestHandlerClass = VP.make_handler(upstream, cls.encoder, {f"http://127.0.0.1:{port}"})
        threading.Thread(target=cls.px.serve_forever, daemon=True).start()
        cls.base, cls.upstream = f"http://127.0.0.1:{port}", upstream

    @classmethod
    def tearDownClass(cls):
        for s in (cls.px, cls.up):
            s.shutdown()
            s.server_close()
        cls.d.cleanup()

    def setUp(self):
        FakeServer.log.clear()
        self.vision.calls.clear()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()

    def sent(self, path):
        return [json.loads(b) for m, p, h, b in FakeServer.log if m == "POST" and p == path]

    def test_openai_parts_encoded_and_uploaded_once(self):
        FakeServer.uploads.clear()                                     # a fresh server: nothing uploaded yet
        msg ={"role": "user", "content": [{"type": "text", "text": "?"},
                                           {"type": "video_url", "video_url": {"url": "C:/a.mp4", "fps": 1}},
                                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}
        out = self.post("/v1/chat/completions", {"model": "m", "messages": [msg], "stream": True})
        self.assertEqual(out.count(b"data:"), 4)                       # the stream came through whole
        self.assertIn(("video", "C:/a.mp4", {"fps": 1}), self.vision.calls)
        parts = self.sent("/v1/chat/completions")[0]["messages"][0]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "video_embeddings", "image_embeddings"])
        self.assertEqual(len([1 for m, p, h, b in FakeServer.log if p == "/v1/strata/embeddings"]), 2)
        FakeServer.log.clear()
        self.post("/v1/chat/completions", {"model": "m", "messages": [msg, {"role": "assistant", "content": "x"}, msg]})
        self.assertEqual([p for m, p, h, b in FakeServer.log if m == "POST"], ["/v1/chat/completions"])  # no re-upload

    def test_anthropic_and_responses(self):
        self.post("/v1/messages", {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}},
            {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "video", "video": "C:/b.mp4"}]}]}]})
        blocks = self.sent("/v1/messages")[0]["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "image_embeddings")
        self.assertEqual(blocks[1]["content"][0]["type"], "video_embeddings")   # nested in a tool result too
        self.post("/v1/responses", {"model": "m", "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "?"}, {"type": "input_video", "video_url": "C:/c.mp4"}]}]})
        items = self.sent("/v1/responses")[0]["input"][0]["content"]
        self.assertEqual([p["type"] for p in items], ["input_text", "video_embeddings"])
        self.post("/v1/responses", {"model": "m", "input": "plain text"})            # a string input passes as it is
        self.assertEqual(self.sent("/v1/responses")[1]["input"], "plain text")

    def test_pass_through_and_origin(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:
            self.assertEqual(r.read(), b"<html>strata</html>")
        self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                  {"Origin": self.base})
        h = [h for m, p, h, b in FakeServer.log if p == "/v1/chat/completions"][0]
        self.assertEqual(h["Origin"], self.upstream.origin)               # our page is the server's page
        self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                  {"Origin": "https://evil.example"})
        h = [h for m, p, h, b in FakeServer.log if p == "/v1/chat/completions"][-1]
        self.assertEqual(h["Origin"], "https://evil.example")             # anyone else's stays: the server refuses

    def test_too_long_an_answer_is_shortened_and_sent_again(self):
        out = self.post("/v1/chat/completions", {"model": "m", "max_tokens": 32000, "stream": True,
                                                 "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn(b"[DONE]", out)                                   # the client got the answer, not the 400
        sent = self.sent("/v1/chat/completions")
        self.assertEqual([s["max_tokens"] for s in sent], [32000, 4512])

    def test_short_answers_skip_thinking(self):
        msgs = [{"role": "user", "content": "should this be blocked?"}]
        self.post("/v1/chat/completions", {"model": "m", "max_tokens": 256, "messages": msgs, "tool_choice": "required"})
        self.post("/v1/chat/completions", {"model": "m", "max_tokens": 256, "messages": msgs, "reasoning_effort": "high"})
        self.post("/v1/chat/completions", {"model": "m", "max_tokens": 4096, "messages": msgs})
        sent = self.sent("/v1/chat/completions")
        self.assertEqual([s.get("reasoning_effort") for s in sent], ["none", "high", None])   # asked for: kept

    def test_videos_in_one_request_share_the_budget(self):
        self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Video A:"}, {"type": "video_url", "video_url": {"url": "C:/target.mp4"}},
            {"type": "text", "text": "Video B:"}, {"type": "video_url", "video_url": {"url": "C:/ours.mp4"}},
            {"type": "text", "text": "What differs?"}]}]})
        self.assertEqual(self.vision.videos_in_request, 2)
        self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(self.vision.videos_in_request, 1)

    def test_video_budget_follows_the_servers_context(self):
        self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "C:/d.mp4"}}]}]})
        self.assertEqual(self.vision.context, 262144)                    # from the server's /health

    def test_encoder_error_is_a_400(self):
        def boom(source, opts):
            raise ValueError("the video could not be read: nope")
        self.vision.encode_video, saved = boom, self.vision.encode_video
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": "x.mp4"}}]}]})
            self.assertEqual(e.exception.code, 400)
            self.assertIn("nope", json.loads(e.exception.read())["error"]["message"])
        finally:
            self.vision.encode_video = saved
        self.assertEqual(self.sent("/v1/chat/completions"), [])           # nothing reached the server


class VocabGguf(unittest.TestCase):
    def test_keys_kept_split_dropped(self):
        import make_vocab_gguf as MV

        def kv(key, t, val):
            return struct.pack("<Q", len(key)) + key + struct.pack("<I", t) + val
        s = lambda b: struct.pack("<Q", len(b)) + b                       # noqa: E731
        src = b"GGUF" + struct.pack("<IQQ", 3, 5, 3) + kv(b"general.architecture", 8, s(b"qwen")) + \
            kv(b"split.count", 4, struct.pack("<I", 2)) + \
            kv(b"tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + s(b"a") + s(b"bc")) + b"tensor info..."
        out, n = MV.vocab_gguf(src)
        self.assertEqual(n, 2)
        self.assertEqual(struct.unpack_from("<IQQ", out, 4), (3, 0, 2))     # no tensors, two keys
        self.assertNotIn(b"split.count", out)
        self.assertIn(b"tokenizer.ggml.tokens", out)


if __name__ == "__main__":
    unittest.main()
