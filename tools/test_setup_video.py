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
                self.assertEqual("CUDA" in out, asked == "gpu")          # a GPU encoder was asked for: say what it needs

    def test_up_to_date_or_no_images(self):
        self.stamp(vision_src=self.vsrc, vision="cpu")
        self.assertEqual(self.call("cpu")[:2], ("cpu", []))
        self.stamp(vision_src="upstream")
        self.assertEqual(self.call("none")[:2], ("none", []))

    def gpu_call(self, cuda, arch=120):
        builds = []
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(setup, "find_nvcc", return_value=("nvcc.exe", cuda)), \
                mock.patch.object(setup, "find_vcvars_cuda", return_value=("vcvars64.bat", "-vcvars_ver=14.44")), \
                mock.patch.object(setup, "cmake_build", side_effect=lambda *a: builds.append(a)), \
                mock.patch.object(setup.shutil, "copy2"), \
                mock.patch.object(setup, "build_vision_cpu", side_effect=lambda *a: builds.append(("cpu",))):
            got = setup.fork_vision(self.eng, "gpu", Path("llama"), {"arch": arch})
        return got, builds

    def test_gpu_encoder_with_a_new_enough_toolkit(self):
        self.stamp(vision_src="upstream")
        got, builds = self.gpu_call((12, 8))
        self.assertEqual(got, "gpu")
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120", builds[0][3])
        self.assertEqual(builds[0][6], "-vcvars_ver=14.44")
        meta = json.loads((self.eng / "BUILD.json").read_text())
        self.assertEqual((meta["vision"], meta["vision_src"]), ("gpu", self.vsrc))
        self.assertEqual(self.gpu_call((12, 8))[1], [])                   # built: not again

    def test_too_old_a_toolkit_for_rtx_50_builds_for_the_cpu(self):
        self.stamp(vision_src="upstream")
        self.assertEqual(self.gpu_call((12, 4)), ("cpu", [("cpu",)]))
        self.assertEqual(self.gpu_call((12, 4), arch=89)[0], "gpu")       # an RTX 40 card takes 12.4

    @unittest.skipUnless(setup.WIN, "Visual Studio is looked for on Windows only")
    def test_no_compiler_keeps_images(self):
        self.stamp(vision_src="upstream")
        got, built, out = self.call("gpu", vcvars=None)
        self.assertEqual((got, built), ("gpu", []))
        self.assertIn("images only", out)


class RemoteVision(unittest.TestCase):
    """--vision remote: a server whose pictures and videos arrive encoded from another PC (tools/video_proxy.py)."""

    def install(self, *extra, configs=()):
        sys.path.insert(0, str(ROOT / "tools"))
        from test_setup_golden import card, install
        found = [card(0, "NVIDIA GeForce RTX 2080 Ti", 11.0, "75")]          # the user's server: 11 GB + 128 GB
        return install(127.8, found, ["--family", "qwen", "--model", "IQ3_S", "--no-start", *extra], configs=configs)

    def test_engine_vision_without_an_encoder(self):
        code, out, cfg, _ = self.install("--vision", "remote")
        self.assertEqual(code, 0, out)
        a = cfg["args"]
        self.assertEqual(a.count("--vision"), 1)
        self.assertNotIn("--vram-reserve-mib", a)                           # no encoder needs VRAM here
        self.assertEqual(cfg["vision"]["remote"], True)
        self.assertNotIn("exe", cfg["vision"])
        self.assertNotIn("vision encoder:", out)                             # no mmproj download
        self.assertIn("video_proxy.py", out)

    def test_choice_kept_and_not_leaked(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-iq3_s.json"
            p.write_text(json.dumps({"args": ["--vision"], "vision": {"remote": True, "upload_dir": "u"}}))
            self.assertEqual(setup.choices_from_config(p)["vision"], "remote")
        new = {"vision": {"exe": "v", "mmproj": "m", "model": "x", "gpu": True, "max_tokens": 1024}}
        with contextlib.redirect_stdout(io.StringIO()):
            setup.carry_over({"vision": {"remote": True, "upload_dir": "u", "video_fps": 1}}, new)
        self.assertNotIn("remote", new["vision"])                            # a local encoder again: not remote
        self.assertEqual(new["vision"]["video_fps"], 1)                      # the user's own keys still carry over


if __name__ == "__main__":
    unittest.main()
