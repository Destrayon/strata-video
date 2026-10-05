# Comparing videos (target gameplay vs our game)

1. **Shared budget** - videos in one request split the automatic budget (server prepare and the proxy count them).
   Done when: tests show two videos each get half.
2. **watch_video paths** - `paths: [...]` (2-4 videos) labelled "Video A: <name>", ... in one conversation.
   Done when: tests pass.
3. **Test** - the user's 51 s recording against a copy with known changes (minimap hidden, colours washed out).
   Done when: the answer and time are recorded.
4. **Docs** - DETAILS.md watch_video section. Done when: committed and pushed.

## Notes

- Phases 1-2: `videos_in_request` divides the automatic budget (server prepare, proxy rewrite); watch_video `paths`
  (3fd5e2e). 156 server, 9 proxy, 4 tool tests pass.
- Phase 3: user's 51 s 1080p recording vs `E:\tmp\strata-tests\variant.mp4` (minimap box, saturation 0.3, 1.25x):
  all three found, one false difference ("Obsidian Stance"), 42.6K tokens, 153 s.
- Phase 4: DETAILS.md. Next action: run it on real target-vs-ours clip pairs.
