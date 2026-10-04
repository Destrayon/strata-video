# Video understanding for Strata

Goal: `video_url` content parts reach Qwen3.8-Flash-Next as llama.cpp's mtmd would feed them - frames sampled by
ffmpeg, consecutive frames merged into one temporal patch, `[XmY.Zs]` timestamp text between frame groups, and
3-D M-RoPE positions (time, height, width) per visual token.

Design: the engine already takes a per-cell (t, h, w) table and embedding rows at `<|image_pad|>` cells (GENI).
A new record `SVE2` carries explicit per-token position offsets and the position advance, so the engine places
video frame groups exactly where mtmd's `mtmd_image_tokens_get_decoder_pos` puts them. Video cells keep the
image-pad token id, as llama.cpp does (its PLE reads `ple.image_token_id` for every embedding-input cell).

## Phases

1. **Fork** - new private repo `Destrayon/strata-video`, upstream remote kept, work on branch `video`.
   Done when: pushed and `git remote -v` shows origin + upstream.
2. **mtmd study** - fetch the pinned llama.cpp (3cf0325), read the video helper and qwen position code.
   Done when: the frame-merge rule, timestamp chunks and video decoder positions are written down below.
3. **Encoder** - `strata-vision` gets `ENCV <video> <out> [fps]`: writes SVE2 records plus a layout line.
   Done when: it builds and encodes a real clip with the model's mmproj.
4. ~~Engine~~ - dropped after phase 2: frame groups are ordinary SVE1 image records (see Notes).
5. **Server** - OpenAI `video_url` / `{"type":"video"}` parts and Anthropic-style video blocks, layout splicing,
   caching by hash, `video` options in the config. Done when: new server tests pass with the mock engine.
6. **Docs** - DETAILS.md video section and README note. Done when: written, with what was measured.

## Notes

- Phase 1: private repo https://github.com/Destrayon/strata-video, `upstream` = Niko1221/Strata, branch `video`.
- Phase 2 (llama.cpp 3cf0325, `tools/mtmd`): a video file passed to `mtmd_helper_bitmap_init_from_file` becomes a
  lazy bitmap; one `mtmd_tokenize` with one marker expands it into text `Video:`, then per frame group
  `<|vision_start|>` + image chunk + `<|vision_end|>`, with `[XmY.YYs]` timestamp text every 5 s (default 4 fps).
  The mmproj is `qwen3vl_merger`, so `clip_model_n_temporal_merge` = 2: two consecutive frames form one chunk.
  Each chunk is an ordinary M-RoPE image (t = p, h = p + y, w = p + x, advance max(nx, ny)) - exactly what the
  engine's SVE1 path already does. **So the engine needs no change: phase 4 dropped, no SVE2 format.**
  `MTMD_VIDEO` is on by default (LLAMA_SUBPROCESS on for desktop); ffmpeg/ffprobe must be on PATH or given.
  Video cells keep the image-pad id, matching llama.cpp (PLE reads `ple.image_token_id` for embedding cells).
- Phase 3: `ENCV <fps> <max_frames> <max_side> <timestamp_ms> <video> <out>` in `tools/vision/strata_vision.cpp`
  (+ `--ffmpeg-dir`). Verified on CPU with llama.cpp's test-3.mp4 (720x358, 10 s): 2 fps / 448 px -> 20 frames,
  11 groups, 1,078 tokens, 6.5 s; cap 6 -> 6 frames, 4 groups, 392 tokens; ENC image unchanged (300 tokens).
  Upstream quirk kept: the timestamp text follows its frame, so frame 0 is unpaired. Test model: a tensor-less
  GGUF built from the IQ2_XS shard-1 header (`vocab-only.gguf`, outside the repository).
  Build: VS 2026's own CMake + Ninja (the pip `ninja` on PATH is broken). Next: the server's video parts.
