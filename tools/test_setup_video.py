"""Tests for setup.py's fork_vision (strata-video): beside a ready-made engine, this fork's own image encoder - the one
that reads videos - is compiled for the CPU, once per source change.  Mocked builds - no GPU, no compiler.

    python -m unittest tools.test_setup_video
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class ForkVision(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.eng = Path(self.d.name)
        self.vsrc = setup.source_hash(setup.VISION_SOURCES)

    def tearDown(self):
        self.d.cleanup()

    def stamp(self, **meta):
        (self.eng / "BUILD.json").write_text(json.dumps({"source": "prebuilt", "version": "0.1.39", **meta}))

    def call(self, vision, vcvars="vcvars64.bat"):
        built = []
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.object(setup, "find_vcvars", return_value=vcvars), \
                mock.patch.object(setup, "build_vision_cpu", side_effect=lambda *a: built.append(a)):
            got = setup.fork_vision(self.eng, vision, Path("llama"))
        return got, built, out.getvalue()

    def test_ready_made_encoder_is_replaced_on_the_cpu(self):
        self.stamp(vision_src="upstream")
        for asked in ("gpu", "cpu"):
            with self.subTest(asked=asked):
                got, built, out = self.call(asked)
                self.assertEqual(got, "cpu")
                self.assertEqual(len(built), 1)
                self.assertEqual(built[0][4], self.vsrc)                 # stamped with this fork's source
                self.assertEqual("--build" in out, asked == "gpu")       # a GPU encoder was asked for: say how

    def test_up_to_date_or_no_images(self):
        self.stamp(vision_src=self.vsrc, vision="cpu")
        self.assertEqual(self.call("cpu")[:2], ("cpu", []))
        self.stamp(vision_src="upstream")
        self.assertEqual(self.call("none")[:2], ("none", []))

    @unittest.skipUnless(setup.WIN, "Visual Studio is looked for on Windows only")
    def test_no_compiler_keeps_images(self):
        self.stamp(vision_src="upstream")
        got, built, out = self.call("gpu", vcvars=None)
        self.assertEqual((got, built), ("gpu", []))
        self.assertIn("images only", out)


if __name__ == "__main__":
    unittest.main()
