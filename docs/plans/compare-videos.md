# Comparing videos (target gameplay vs our game)

1. **Shared budget** - videos in one request split the automatic budget (server prepare and the proxy count them).
   Done when: tests show two videos each get half.
2. **watch_video paths** - `paths: [...]` (2-4 videos) labelled "Video A: <name>", ... in one conversation.
   Done when: tests pass.
3. **Test** - the user's 51 s recording against a copy with known changes (minimap hidden, colours washed out).
   Done when: the answer and time are recorded.
4. **Docs** - DETAILS.md watch_video section. Done when: committed and pushed.

## Notes
