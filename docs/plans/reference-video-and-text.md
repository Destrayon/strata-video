# Reference Qwen3-VL video format and readable text

Research: [docs/research/video-game-qa-vlm-2026-10-04.md](../research/video-game-qa-vlm-2026-10-04.md).

1. **Baseline** - a 1080p test clip with small changing on-screen text (HUD numbers, a subtitle line, a timer);
   ask the running model to read it at today's settings. Done when: read-out accuracy and timing error recorded.
2. **Encoder** - `strata-vision` ENCV builds the reference prompt itself: `<t.t seconds>` before every frame pair
   (the pair's mean time), pairs (0,1), (2,3)..., no "Video:", an odd last frame duplicated; frames sized by a
   token budget per pair instead of a fixed side. Done when: it builds and the layout shows that structure.
3. **Server** - a per-pair `tokens` option and default, `max_side` kept as an extra cap, tests updated.
   Done when: server tests pass.
4. **Measure** - GPU encoder rebuilt and installed, the text clip and the barbershop clip asked again.
   Done when: before/after numbers recorded.
5. **Docs** - DETAILS.md video section updated. Done when: committed and pushed.

## Notes

- Phase 1: test clips drawn with Pillow (ffmpeg's drawtext crashes here: no fontconfig) in the scratchpad,
  `text-hud.mp4` (clean) and `text-hud-hard.mp4` (busy background, HUD 20 px, sign 18 px, subtitles 28 px), scored by
  `text_eval.py`. Hard clip before: defaults 0/3 subtitles, 1/4 HUD, sign wrong; 1 fps / 896 px 3/3, 3/4, sign right.
  Clean clip: timing off by ~1 s at both settings (the timestamp mismatch).
