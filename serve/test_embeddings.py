"""serve/test_embeddings.py - pictures and videos encoded on another PC (strata-video): the bundle format, the store,
the image_embeddings / video_embeddings parts and the upload endpoint, against the mock engine (no GPU).

    python -m unittest serve.test_embeddings -v
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import embeddings as E  # noqa: E402
from serve.frontend import EMBEDDINGS_REF, ChatTemplate, openai_to_messages, videos_of  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

W = 8                                                   # a narrow embedding width keeps the files tiny


def sve(*grids, seed=0):
    """strata-vision's output: one f32 SVE1 record per (nx, ny)."""
    rng = np.random.default_rng(seed)
    out = b""
    for nx, ny in grids:
        out += struct.pack("<5i", E.RECORD, nx * ny, nx, ny, W) + rng.standard_normal(nx * ny * W).astype("<f4").tobytes()
    return out


def rows(data):
    off, out = 0, []
    while off < len(data):
        _, n, _, _, w = struct.unpack_from("<5i", data, off)
        out.append(np.frombuffer(data, "<f4", n * w, off + 20))
        off += 20 + n * w * 4
    return np.concatenate(out)


VIDEO_LAYOUT = [[70, 71, 900], 2, [901, 72, 900], 4, [901]]


class Bundle(unittest.TestCase):
    def test_round_trip(self):
        raw = sve((2, 1), (2, 2))
        for dtype, tol in (("f32", 0), ("f16", 2e-3)):
            with self.subTest(dtype=dtype):
                b = E.make_bundle("video", raw, 6, VIDEO_LAYOUT, dtype, "abc")
                h, back = E.parse_bundle(b)
                self.assertEqual((h["kind"], h["n"], h["n_embd"], h["layout"]), ("video", 6, W, VIDEO_LAYOUT))
                self.assertEqual(len(back), len(raw))                     # f32 again, as the engine reads it
                np.testing.assert_allclose(rows(back), rows(raw), atol=tol, rtol=tol)
                if dtype == "f16":                                         # half the upload, at a real size
                    big = sve((28, 16))
                    self.assertLess(len(E.make_bundle("image", big, 448, None, "f16")), len(big) * 0.55)

    def test_malformed(self):
        raw = sve((2, 1), (2, 2))
        good = E.make_bundle("video", raw, 6, VIDEO_LAYOUT, "f32")
        bad = {
            "magic": b"XXXX" + good[4:],
            "short": good[:-4],
            "n": E.make_bundle("video", raw, 7, VIDEO_LAYOUT, "f32"),
            "layout": E.make_bundle("video", raw, 6, [[1], 4, [2], 2], "f32"),
            "no layout": E.make_bundle("video", raw, 6, None, "f32"),
            "image of two": E.make_bundle("image", raw, 6, None, "f32"),
        }
        for name, data in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                E.parse_bundle(data)


class Store(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.store = E.EmbeddingStore(Path(self.d.name), n_embd=W, mmproj_sha256="enc1")

    def tearDown(self):
        self.d.cleanup()

    def test_put_get(self):
        b = E.make_bundle("image", sve((3, 2)), 6, None, "f16", "enc1")
        r = self.store.put(b)
        self.assertEqual((r["kind"], r["n"]), ("image", 6))
        self.assertTrue(self.store.has(r["id"]))
        path, n = E.lookup(self.store, EMBEDDINGS_REF + r["id"], "image")
        self.assertEqual(n, 6)
        self.assertEqual(path.read_bytes()[:4], struct.pack("<i", E.RECORD))
        self.assertIsNone(E.lookup(self.store, "C:/clip.mp4", "video"))  # not a reference: the encoder's job
        with self.assertRaisesRegex(ValueError, "not a video"):
            self.store.get(r["id"], "video")
        with self.assertRaisesRegex(ValueError, "no embeddings"):
            self.store.get("0" * 64, "image")
        with self.assertRaises(ValueError):
            self.store.get("../../etc/passwd", "image")

    def test_wrong_model(self):
        other = E.EmbeddingStore(Path(self.d.name) / "w", n_embd=W * 2)
        with self.assertRaisesRegex(ValueError, "wide"):
            other.put(E.make_bundle("image", sve((1, 1)), 1, None, "f32"))
        with self.assertRaisesRegex(ValueError, "another vision encoder"):
            self.store.put(E.make_bundle("image", sve((1, 1)), 1, None, "f32", "enc2"))

    def test_oldest_go_first(self):
        small = E.EmbeddingStore(Path(self.d.name) / "s", max_bytes=1)
        a = small.put(E.make_bundle("image", sve((1, 1), seed=1), 1, None, "f32"))["id"]
        b = small.put(E.make_bundle("image", sve((1, 1), seed=2), 1, None, "f32"))["id"]
        self.assertFalse(small.has(a))
        self.assertTrue(small.has(b))                                    # the one just sent stays


class RemoteServer(unittest.TestCase):
    """A server in "vision": {"remote": true} mode: uploads, parts and refusals, over HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        cls.tok = ByteTokenizer()
        store = E.EmbeddingStore(Path(cls.d.name) / "uploads")
        cls.vision = E.RemoteVision(store, Path(cls.d.name))
        cls.svc = Service(MockEngine(cls.tok, "ok", max_context=8192), cls.tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=cls.vision)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.d.cleanup()

    def http(self, method, path, body=None, ctype="application/json"):
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_upload_then_ask(self):
        b = E.make_bundle("video", sve((2, 1), (2, 2)), 6, VIDEO_LAYOUT, "f16")
        s, r = self.http("POST", "/v1/strata/embeddings", b, "application/octet-stream")
        self.assertEqual((s, r["kind"], r["n"]), (200, "video", 6), r)
        self.assertEqual(self.http("GET", f"/v1/strata/embeddings/{r['id']}")[0], 200)
        self.assertEqual(self.http("GET", "/v1/strata/embeddings/" + "1" * 64)[0], 404)
        img = self.http("POST", "/v1/strata/embeddings", E.make_bundle("image", sve((3, 1)), 3, None, "f16"),
                        "application/octet-stream")[1]
        content = [{"type": "text", "text": "what happens?"}, {"type": "video_embeddings", "id": r["id"]},
                   {"type": "image_embeddings", "id": img["id"]}]
        s, out = self.http("POST", "/v1/chat/completions", {"model": "m", "max_tokens": 4,
                                                            "messages": [{"role": "user", "content": content}]})
        self.assertEqual(s, 200, out)
        # the prompt the engine got: the video's layout and 6 + 3 image cells
        msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": content}]})
        ids, _, _ = self.svc.prepare(msgs, None, {})
        pad = self.tok.encode("<|image_pad|>", parse_special=True)[0]
        self.assertEqual(ids.count(pad), 9)
        self.assertEqual(videos_of(msgs), [(EMBEDDINGS_REF + r["id"], {})])

    def test_refusals(self):
        s, out = self.http("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "https://x/a.mp4"}}, {"type": "text", "text": "?"}]}]})
        self.assertEqual(s, 400)
        self.assertIn("video proxy", out["error"]["message"])
        s, out = self.http("POST", "/v1/strata/embeddings", b"SVB1junk", "application/octet-stream")
        self.assertEqual(s, 400)
        s, out = self.http("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "video_embeddings", "id": "2" * 64}, {"type": "text", "text": "?"}]}]})
        self.assertEqual(s, 400)
        self.assertIn("upload them first", out["error"]["message"])

    def test_health_says_videos(self):
        with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
            h = json.loads(r.read())
        self.assertTrue(h["images"] and h["videos"])


if __name__ == "__main__":
    unittest.main()
