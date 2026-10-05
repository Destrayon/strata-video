#!/usr/bin/env python3
"""tools/video_mcp.py - an MCP server with one tool, watch_video: an AI assistant asks Strata about a video file.

    Qwen Code (~/.qwen/settings.json):  "mcpServers": {"strata-video": {"command": "python",
                                         "args": ["<Strata>/tools/video_mcp.py"], "timeout": 3600000}}
    Claude Code:                        claude mcp add strata-video -- python <Strata>/tools/video_mcp.py

The assistant passes a local video path and a question; the video goes to Strata as a `video_url` with that path -
through tools/video_proxy.py (default http://127.0.0.1:8090), which encodes the file from disk at full quality and
fits it to the model's context - and the assistant gets the model's answer as text.  So a coding agent can ask about a
video of any size (Qwen Code's own file reading stops at 10 MB), and the video never enters the agent's own context:
it costs the agent a question and an answer.  Questions about the same file continue one conversation, so the server
reuses the video it has read (seconds instead of minutes) while that conversation is still cached.

MCP (JSON-RPC 2.0, one message per line) over stdin/stdout, Python standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

PROTOCOL = "2025-06-18"
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
VERSION = "0.1.0"
VIDEO_EXT = (".mp4", ".m4v", ".webm", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".ogv", ".ts")
EFFORTS = ("none", "low", "medium", "high")
INSTRUCTIONS = ("watch_video asks the Strata model about a local video file and returns its answer. The model sees the "
                "whole video with each moment's time, so ask directly (what happens, when, what is on screen) - no "
                "need to extract frames or cut the video. Follow-up questions about the same file are fast.")

TOOL = {
    "name": "watch_video",
    "title": "Watch a video",
    "description": "Ask the Strata vision model a question about a local video file (any size or length). The model "
                   "watches the whole video - frames, motion, on-screen text, each moment's time in seconds - and "
                   "answers in text. Use it instead of reading the video file or extracting frames. Ask follow-up "
                   "questions about the same path freely: they reuse the video already read.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path of the video file on this PC"},
            "question": {"type": "string", "description": "What to find out about the video"},
            "new_conversation": {"type": "boolean", "description": "Start over instead of continuing the "
                                 "conversation about this file (default false)"},
            "effort": {"type": "string", "enum": list(EFFORTS),
                       "description": "How much the model thinks before answering (default low)"},
            "fps": {"type": "number", "description": "Frames per second to sample, at most (default 2)"},
            "tokens": {"type": "integer", "description": "Tokens per frame pair; 1024 reads small text best "
                       "(default 448)"},
            "total_tokens": {"type": "integer", "description": "The whole video's token budget (default: "
                             "automatic, 60% of the model's context)"},
        },
        "required": ["path", "question"],
    },
}


class ToolError(Exception):
    pass


class Watcher:
    """Conversations per video file, sent to Strata's OpenAI API (the video proxy)."""

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 3600):
        self.url, self.api_key, self.timeout = url.rstrip("/"), api_key, timeout
        self.chats: dict[str, list] = {}
        self.lock = threading.Lock()

    def call(self, args) -> dict:
        if not isinstance(args, dict):
            raise ToolError("arguments must be an object")
        path, question = args.get("path"), args.get("question")
        if not isinstance(path, str) or not path.strip() or not isinstance(question, str) or not question.strip():
            raise ToolError("path and question are required")
        path = os.path.abspath(os.path.expanduser(path.strip()))
        if not os.path.isfile(path):
            raise ToolError(f"no such file: {path}")
        if not path.lower().endswith(VIDEO_EXT):
            raise ToolError(f"not a video file (by its extension): {path}")
        effort = args.get("effort") or "low"
        if effort not in EFFORTS:
            raise ToolError(f"effort must be one of {', '.join(EFFORTS)}")
        opts = {k: args[k] for k in ("fps", "tokens", "total_tokens") if args.get(k) is not None}
        key = os.path.normcase(path) + json.dumps(opts, sort_keys=True)
        with self.lock:
            if args.get("new_conversation"):
                self.chats.pop(key, None)
            history = list(self.chats.get(key, []))
        if not history:
            history = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": path, **opts}},
                                                    {"type": "text", "text": question.strip()}]}]
        else:
            history.append({"role": "user", "content": question.strip()})
        body = {"model": "strata", "messages": history, "reasoning_effort": effort, "max_tokens": 4096}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", {}).get("message")
            except (ValueError, AttributeError):
                msg = None
            raise ToolError(f"Strata refused the request: {msg or e}") from None
        except (OSError, ValueError) as e:
            raise ToolError(f"cannot reach Strata at {self.url} ({e}); is tools/video_proxy.py running?") from None
        answer = (out["choices"][0]["message"].get("content") or "").strip()
        with self.lock:
            self.chats[key] = history + [{"role": "assistant", "content": answer}]
        usage = out.get("usage") or {}
        return {"answer": answer, "video": path, "follow_up": len(history) > 1,
                "prompt_tokens": usage.get("prompt_tokens"), "seconds": round(time.monotonic() - t0, 1)}


class Server:
    def __init__(self, watcher: Watcher, out):
        self.watcher, self.out = watcher, out
        self.write_lock = threading.Lock()
        self.protocol = PROTOCOL

    def send(self, msg: dict) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self.write_lock:
            self.out.write(data)
            self.out.flush()

    @staticmethod
    def result(rid, result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    @staticmethod
    def error(rid, code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    def content(self, res: dict, error: bool) -> dict:
        text = res["answer"] if "answer" in res and not error else json.dumps(res, ensure_ascii=False)
        out = {"content": [{"type": "text", "text": text}], "isError": error}
        if self.protocol >= "2025-06-18":
            out["structuredContent"] = res
        return out

    def handle(self, msg) -> dict | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self.error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
        if "method" not in msg or "id" not in msg:
            return None
        method, rid, params = msg["method"], msg["id"], msg.get("params") or {}
        if method == "initialize":
            asked = str(params.get("protocolVersion") or PROTOCOL)
            self.protocol = asked if asked in PROTOCOLS else PROTOCOL
            return self.result(rid, {"protocolVersion": self.protocol,
                                     "capabilities": {"tools": {"listChanged": False}},
                                     "serverInfo": {"name": "strata-video", "title": "Strata video",
                                                    "version": VERSION},
                                     "instructions": INSTRUCTIONS})
        if method == "ping":
            return self.result(rid, {})
        if method == "tools/list":
            return self.result(rid, {"tools": [TOOL]})
        if method == "tools/call":
            if params.get("name") != TOOL["name"]:
                return self.error(rid, -32602, f"unknown tool: {params.get('name')}")
            try:
                return self.result(rid, self.content(self.watcher.call(params.get("arguments")), error=False))
            except ToolError as e:
                return self.result(rid, self.content({"error": str(e)}, error=True))
            except Exception as e:                      # noqa: BLE001 - a bug must not end the server
                return self.result(rid, self.content({"error": f"internal error: {e}"}, error=True))
        if method in ("resources/list", "prompts/list"):
            return self.result(rid, {method.split("/")[0]: []})
        return self.error(rid, -32601, f"method not found: {method}")

    def _run(self, msg):
        res = self.handle(msg)
        if res is not None:
            self.send(res)

    def serve(self, inp) -> None:
        for raw in inp:
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send(self.error(None, -32700, "parse error"))
                continue
            if isinstance(msg, dict) and msg.get("method") == "tools/call":   # minutes long: pings still answered
                threading.Thread(target=self._run, args=(msg,), daemon=True).start()
            else:
                self._run(msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("STRATA_VIDEO_URL", "http://127.0.0.1:8090"),
                    help="Strata's address: the video proxy (default) or a server with its own encoder")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY"),
                    help="only when talking to a server with a key directly (the proxy adds its own)")
    a = ap.parse_args(argv)
    Server(Watcher(a.url, a.api_key), sys.stdout.buffer).serve(sys.stdin.buffer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
