# GPU video encoder and a video button in the web page

1. **CUDA 12.8** - install the toolkit only (nvcc, runtime, cuBLAS, VS integration; the driver stays).
   Done when: `nvcc --version` says 12.8.
2. **Web page video** - attach a video in the chat page, sent as a `video_url` part.
   Done when: a video attached in the browser gets an answer about it.
3. **GPU encoder** - build `strata-vision` with CUDA 12.8 for sm_120 (VS 2026 needs `-allow-unsupported-compiler`),
   teach setup to do the same when such a toolkit is there, switch the config to the GPU encoder.
   Done when: the server encodes the test clip on the GPU and the time is measured.
4. **Docs** - DETAILS.md video section updated with the GPU numbers and the web button.
   Done when: committed and pushed.

## Notes

- Phase 1: CUDA 12.8.1 network installer, packages nvcc/cudart/cublas/cublas_dev/thrust/nvtx/VS integration (`crt`,
  `nvvm`, `nvptxcompiler` are not 12.8 package names - the first run failed on them). nvcc V12.8.93; driver 595.97 kept.
- Phase 2: `serve/web/app.js` (videos via the attach button, drop, data: URL in a `video_url` part, `<video>` in the
  message), `app.css`, `sprite.svg` (`i-video`), `/health` "videos". Verified in the browser: a dropped clip answered.
- Phase 3 finding: nvcc 12.8 + VS 2026 fails even with `-allow-unsupported-compiler` (cudafe++ access violation); a
  VS 2022 (v143) toolset is needed.
- Phase 3 so far: setup's `fork_vision` builds a GPU encoder when nvcc (12.8+ for sm_120) and `find_vcvars_cuda`
  (VS 2019/2022, or VS 2026 + a 14.3x/14.4x toolset via -vcvars_ver) are found; 5 setup tests pass (f6d4802).
  Blocked: the VS installer (exit 8006) needs Visual Studio closed (a project was open in it), and 14.43
  is out of support - add `Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64` instead. Next: that install,
  then `START-HERE.bat --setup` rebuilds the encoder for the GPU.
- Phase 3 done: VS 2026 + toolset 14.44 (14.43 is out of support; VS must be closed to modify it), setup built the
  GPU encoder after one fix (nvcc path with forward slashes, 2a64929); config now `gpu: true`, `max_tokens: 1024`.
  Encode 0.74 s vs 6.5 s CPU; whole request 6.4 s vs 14.0 s.
- Phase 4: DETAILS.md - setup requirements, web button, CPU/GPU table, text-in-video note, GPU answer timings.
