# A branch for upstream: video understanding only

Built in a worktree (`E:\CodeProject\Strata-upstream`) from `upstream/main` (Niko1221/Strata, history rewritten
since the fork, so the commits are re-applied, not merged). Branch `upstream/video-understanding`, pushed to the fork;
no pull request opened.

In: strata-vision ENCV (Qwen3-VL layout, length-based budget, batches), video parts in the API (OpenAI, Responses,
Anthropic), the budget shared by a request's videos, the chat page's video attach, docs (upstream wording), tests;
setup's broken-PATH-tool fix as its own commit.
Out (fork only): remote encoding / proxy / watch_video, fork_vision, find_vcvars_cuda, plans, research.

1. **Worktree and encoder** - port `tools/vision/strata_vision.cpp`. Done when: it builds.
2. **Server and web page** - port frontend/responses/server video handling and app.js/css/sprite. Done when: upstream's
   and our tests pass.
3. **Docs and setup fix** - DETAILS.md/README in upstream wording; find_tool commit. Done when: committed.
4. **Verify and push** - encoder run on a real clip, the server end to end if the installed engine allows; push.
   Done when: pushed, with what was and was not verified written down.

## Notes

- Upstream's history was rewritten after the fork (the fork's base is not in it), so the work was re-applied by hand
  onto upstream/main 82f46a8. Upstream's own image hardening is kept for videos (network paths refused, URL size cap
  read at call time, local files only from trusted origins, names relative to the encoder's dir, combined file after
  the context check).
- Branch `pr/video-understanding` on the fork (worktree `E:\CodeProject\Strata-upstream`), 5 commits, 11 files,
  +664/-37: 47579ce encoder, 0e2d431 serve, 50a8ade web, 030c225 docs, 10eae33 setup fix.
- Verified: encoder builds (CPU) and encodes a picture and a video; upstream's 213 server tests (11 new) and all setup
  tests pass; upstream's server with the port, IQ2_XS on the 5070 Ti, engine 0.1.39 (upstream expects 0.1.40): the
  barbershop revolver at 6 s right; the 36 s HUD video's first HP change right (87 -> 41 at 6 s; the other four not
  named). Not verified on upstream's code: the GPU encoder build, AMD/SYCL, Linux.
- No pull request opened.
