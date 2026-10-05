# Long videos in one pass

A video gets as much of the server's context as a share allows, sized from its real duration: 2 fps at up to 448
tokens per frame pair while that fits, then smaller frames (down to 128 tokens per pair, the Qwen3-VL minimum), then
a lower frame rate. Encoding runs in batches so an hour-scale video does not hold every preprocessed frame at once.
Qwen recommends up to ~224K video tokens for hour-scale video (Qwen3.8-Flash-Next card); research:
[docs/research/video-game-qa-vlm-2026-10-04.md](../research/video-game-qa-vlm-2026-10-04.md).

1. **Encoder** - `strata-vision` ENCV takes the whole-video budget and the per-pair minimum, plans fps / frame size
   from the probed duration, and tokenizes + encodes in batches of pairs. Done when: a 36 s and a ~10 min video
   encode to their planned budgets with bounded memory.
2. **Server and proxy** - the budget follows the context (`video_context_share`, default 0.6, capped at 224K); the
   proxy learns the server's context from `/health`; frame cap raised to Qwen's 2,048. Done when: tests pass.
3. **Install and measure** - GPU encoder rebuilt, the proxy restarted on it, the 36 s video asked through Qwen Code
   in one read, a ~10 min video asked about moments in it. Done when: answers and times recorded.
4. **Qwen Code and docs** - QWEN.md without the manual chunking, DETAILS.md updated. Done when: committed and pushed.

Later (not in this plan): videos longer than the context split automatically in the proxy and the answers merged.

## Notes

- Phase 1: ENCV `<fps> <max_frames> <max_side> <pair_tokens> <total_tokens> <min_pair_tokens>`, plan from the probed
  frame count (3% under budget), batches of 64 frames. GPU encoder, peak RAM flat at 859 MB: 36 s at 39K budget ->
  72 frames / 16,128 tokens / 8.6 s; 10 min at 39K -> 614 frames (~1 fps) / 36,840 / 21 s; 10 min at 157K -> 1,200
  frames (2 fps) / 158,400 / 71 s (0.7% over before the 3% margin). Test video: scratchpad `long-10min.mp4` (20
  scenes of 30 s, "BOSS SPAWNED" 7:13-7:15).
