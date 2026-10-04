# Encode on the PC, run the model on the server

The PC (RTX 5070 Ti) runs `strata-vision`; the server only runs the language model. A local proxy on the PC encodes
every image/video part, uploads the embeddings, rewrites the part to a reference, and forwards the request.

1. **Server** - an embeddings store and `POST /v1/strata/embeddings` (a bundle: JSON header + SVE1 records, f16 or
   f32), `image_embeddings` / `video_embeddings` content parts, and `"vision": {"remote": true}` (no encoder
   process; the engine still runs `--vision`). Done when: server tests for the bundle, the parts and the remote mode
   pass.
2. **Proxy** - `tools/video_proxy.py`: local strata-vision, encodes OpenAI / Anthropic / Responses image and video
   parts, uploads each bundle once (it asks the server first), streams the answer back, passes every other path
   through (web page included); plus `tools/make_vocab_gguf.py` for a PC without the model. Done when: its tests pass
   against a fake server.
3. **Setup** - `--vision remote` for the server (no mmproj, no encoder build, no ffmpeg). Done when: setup tests pass.
4. **End to end** - this PC's server switched to remote mode, the proxy in front of it, the HUD clip asked through
   the API and through the web page. Done when: answers match the local-encoder ones, sizes and times recorded.
5. **Docs** - DETAILS.md section on remote encoding. Done when: committed and pushed.

## Notes

- Server: RTX 2080 Ti (11 GB, sm_75) + 128 GB DDR4, Linux. Remote vision frees the encoder's ~1.4 GB VRAM reserve;
  128 GB RAM allows IQ3_S (matches BF16 on the card's benchmarks).
- Phase 1: `serve/embeddings.py` (bundle, EmbeddingStore, lookup, RemoteVision), server.py (`upload_store`, store in
  Vision, `POST /v1/strata/embeddings`, `GET /v1/strata/embeddings/<id>`, `"remote": true`), frontend/responses parts.
  `serve/test_embeddings.py` 8 tests + 148 server tests pass.
