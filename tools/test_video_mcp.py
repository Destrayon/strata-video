"""tools/test_video_mcp.py - the watch_video MCP server against a fake Strata (no GPU, no network): the protocol, the
checks on its arguments, follow-up questions continuing one conversation, and errors as tool errors.

    python -m unittest tools.test_video_mcp -v
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import video_mcp as VM  # noqa: E402


class FakeStrata(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeStrata.seen.append(req)
        if "refuse" in json.dumps(req["messages"][-1]):
            body, code = {"error": {"message": "the video could not be read"}}, 400
        else:
            body, code = {"choices": [{"message": {"content": f"answer {len(FakeStrata.seen)}"}}],
                          "usage": {"prompt_tokens": 1234}}, 200
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class VideoMcp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeStrata)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.d = tempfile.TemporaryDirectory()
        cls.video = Path(cls.d.name) / "clip.mp4"
        cls.video.write_bytes(b"\0")
        cls.watcher = VM.Watcher(f"http://127.0.0.1:{cls.httpd.server_address[1]}")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.d.cleanup()

    def setUp(self):
        FakeStrata.seen.clear()
        self.watcher.chats.clear()

    def rpc(self, *msgs):
        out = io.BytesIO()
        srv = VM.Server(self.watcher, out)
        for m in msgs:
            srv._run(m)                                                 # in this thread: the answers in order
        return [json.loads(x) for x in out.getvalue().decode().splitlines()]

    def test_protocol(self):
        init, tools = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                               {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(init["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual([t["name"] for t in tools["result"]["tools"]], ["watch_video"])

    def test_follow_up_continues_the_conversation(self):
        a = self.watcher.call({"path": str(self.video), "question": "what happens?", "fps": 1})
        b = self.watcher.call({"path": str(self.video), "question": "and at the end?", "fps": 1})
        self.assertEqual((a["follow_up"], b["follow_up"]), (False, True))
        first, second = FakeStrata.seen
        self.assertEqual(first["messages"][0]["content"][0]["video_url"], {"url": str(self.video), "fps": 1})
        self.assertEqual([m["role"] for m in second["messages"]], ["user", "assistant", "user"])  # the video once
        self.assertEqual(second["messages"][1]["content"], "answer 1")
        self.assertEqual(first["reasoning_effort"], "low")
        c = self.watcher.call({"path": str(self.video), "question": "again", "fps": 1, "new_conversation": True})
        self.assertFalse(c["follow_up"])

    def test_compare_labels_each_video(self):
        other = Path(self.d.name) / "target.webm"
        other.write_bytes(b"\0")
        r = self.watcher.call({"paths": [str(other), str(self.video)], "question": "what differs?"})
        parts = FakeStrata.seen[0]["messages"][0]["content"]
        self.assertEqual([p.get("text") or p["video_url"]["url"] for p in parts],
                         ["Video A: target.webm", str(other), "Video B: clip.mp4", str(self.video), "what differs?"])
        self.assertEqual(r["videos"], [str(other), str(self.video)])
        self.watcher.call({"paths": [str(other), str(self.video)], "question": "and the HUD?"})
        self.assertEqual(len(FakeStrata.seen[1]["messages"]), 3)          # a follow-up of the comparison
        with self.assertRaisesRegex(VM.ToolError, "2-4"):
            self.watcher.call({"paths": [str(self.video)] * 5, "question": "?"})

    def test_bad_arguments_and_refusals_are_tool_errors(self):
        for args, msg in (({"path": str(self.video)}, "required"),
                          ({"path": "Z:/nope.mp4", "question": "?"}, "no such file"),
                          ({"path": __file__, "question": "?"}, "not a video"),
                          ({"path": str(self.video), "question": "?", "effort": "max"}, "effort")):
            with self.subTest(msg), self.assertRaisesRegex(VM.ToolError, msg):
                self.watcher.call(args)
        (res,) = self.rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "watch_video", "arguments": {"path": str(self.video), "question": "refuse"}}})
        self.assertTrue(res["result"]["isError"])
        self.assertIn("could not be read", res["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
