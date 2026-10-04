# Best local model, video settings and pipeline for game-QA video on an RTX 5070 Ti PC
**Date:** 2026-10-04
**Status:** Reviewed
**Built on:** no prior corpus material

## Executive summary
No source shows that any model is good at catching game glitches in video, and no game-QA benchmark has tested
Qwen3.8-Flash-Next. On general video benchmarks, Flash-Next has one official score: LVBench 76.6. That is the best
long-video score among models this PC can run. The best-documented alternative is Qwen3.6-35B-A3B (Video-MME 82.5
without subtitles, MVBench 74.6). The research identifies three levers for game QA:

- Make our video preprocessing match the reference Qwen3-VL scheme. Ours currently differs in timestamps, frame
  pairing and resolution.
- Move from the 2-bit IQ2_XS quantization to the 3-bit IQ3_XXS.
- Build a pipeline in which the model is a high-recall filter for single-frame glitches, compared against a clean
  reference frame, after scene-change prefiltering. Glitches that only show across several frames should be caught
  by deterministic checks instead.

Key uncertainty: every video conclusion about quantization and preprocessing is extrapolated, not measured on video.

## Research brief
**Question:** For glitch and bug detection in gameplay video on a local RTX 5070 Ti (16 GB) PC with 64 GB RAM and
a llama.cpp-based stack, which model, video settings and pipeline work best?

**Sub-questions:**
1. Which open-weight vision-language models are best at video, and which run locally?
2. Is the vision encoder swappable? Does our llama.cpp video path match the reference Qwen3-VL preprocessing?
3. What does the game-QA research show about models, glitch types, false positives, sampling and pipelines?
4. How much does 2-bit quantization of the language model cost on vision and video tasks?

**Out of scope:** closed models as a deployment option (they are cited only for comparison), fine-tuning, and engine
speed.

**Success criteria:** a concrete local recommendation (model, quantization, settings, pipeline) with each point
traced to a source.

## Findings by sub-question

### 1. Video ability of open models that fit 16 GB VRAM + 64 GB RAM
Qwen3.8-Flash-Next's official card lists only one video score: **LVBench 76.6**. The same card gives Qwen3.8-27B
72.4, Qwen3.7-Plus 76.2 and Claude Opus 4.6 Max 63.0 on LVBench, which tests long videos. It lists no Video-MME,
MVBench or MLVU scores, and neither does the ISTA-DASLab GGUF card (primary). Of the models that run on this PC:

| Model | Size | Official video scores | Runs here? |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | 125B total, 6B active (MoE) | LVBench 76.6 only | Yes, at IQ2_XS–IQ3_S |
| Qwen3.6-35B-A3B | 35B total, 3B active (MoE) | Video-MME 86.6 with subtitles / 82.5 without; MLVU 86.2; MVBench 74.6; VideoMMMU 83.7; LVBench 71.4 | Yes, about 21 GB at 4-bit with experts in RAM (estimate) |
| Qwen3.6-27B | 27B dense | Video-MME 87.7 with subtitles; MLVU 86.6; MVBench 75.5; VideoMMMU 84.4 | Tight, about 16–17 GB at 4-bit (estimate) |
| Nemotron 3 Nano Omni | 31B total, 3B active | Video-MME 72.2 | Probably; GGUF support unverified |
| Gemma 4 31B | 31B dense | Card says text and image only | Yes; llama.cpp can still feed it frames as images |

Kimi K2.5 (Video-MME 87.4; 1T parameters), MiMo-V2.5 (310B; its 87.7 rests on search snippets only) and MiniMax M3
(about 428B; aggregator and blog figures conflict) do not fit. llama.cpp added video input through ffmpeg in PR
#24269 (2026-06-08). It works for any supported vision model by expanding the video into frames. Confidence: high
for the scores taken from official cards; low for MiMo and MiniMax.

### 2. The encoder is fixed; our preprocessing differs from the Qwen3-VL reference
The vision encoder cannot be swapped. It is a SigLIP-2 encoder that Qwen trained further, and its projector maps
into this specific model's hidden size, so the mmproj file belongs to its model (Qwen3-VL tech report, arXiv
2511.21631). The table compares our current Strata/llama.cpp path with the reference Qwen3-VL code (transformers
`video_processing_qwen3_vl.py`, `processing_qwen3_vl.py`, vLLM `qwen3_vl.py`, and qwen-vl-utils):

| Aspect | Reference Qwen3-VL | Strata (llama.cpp 3cf0325 mtmd helper) |
| --- | --- | --- |
| Frame rate | 2 fps; 4 fps for grounding; 4–768 frames | 2 fps; 32-frame cap |
| Timestamps | Before every frame pair, `<2.2 seconds>` (the pair's mean time) | `[0m5.00s]` every 5 s, placed after the frame |
| Pairing | (0,1), (2,3), … | Frame 0 alone, then (1,2), (3,4), … |
| Tokens per pair | 128–768 (qwen-vl-utils); transformers spreads ~12,288 tokens across the clip | ~98–112 (448 px) |
| Positions, deepstack, odd final frame | — | Match the reference |
| Prefix text | None | "Video:" |

The tech report says each temporal patch is prefixed with its timestamp, so the model learned to read a timestamp as
describing the frames that follow it. Our timestamps describe the frames before them, and appear 10× less often.
That is consistent with the "about 7 s" answer for an event at about 6 s. Confidence: high on the differences, which
were read from source code. The accuracy cost is an inference: no published llama.cpp-versus-transformers video
comparison exists. We assume Flash-Next uses the Qwen3-VL processor because its projector is `qwen3vl_merger`; this
is not confirmed.

### 3. Game-QA research: models catch single-frame glitches but are near chance on temporal ones
- **Single-frame (spatial) glitches are partly within reach.** On VideoGameQA-Bench (NeurIPS 2025), GPT-4o
  scored 82.8% on image glitch detection and the best open model, Qwen2.5-VL-72B, 70.0%. On video glitch detection,
  Gemini-2.5-Pro scored 78.1% and Qwen2.5-VL-72B 47.9%.
- **Temporal glitches are near chance.** On TempGlitch (2026 preprint), all 12 models scored near 50% on glitches
  that only appear across frames: blinking, frozen animation, abnormal velocity, characters stuck in place.
  Qwen3.6-27B reached about 52 F1. Raising the frame rate from 1 to 5 fps did not reliably help.
- **The model should be a filter, not the judge.** A 2026 clipping-detection study with frames at 6 Hz, judged one
  at a time, found the best open models (Ministral-3-14B, Gemma-4-31B) at about 61–68% accuracy with more than 80%
  recall. Their precision swung widely with prompt wording.
- **False alarms are the practical problem.** VideoGameQA-Bench notes GPT-4.1's false positives would "overwhelm
  human testers".
- **A clean reference frame was the biggest measured gain.** In RefGlitch-Bench, showing Qwen3-VL-8B the last clean
  frame raised its F1 from 0.42 to 0.74.
- **Scene-change prefiltering scales it up.** In EA's industrial study, ffmpeg scene-change detection cut 41 hours
  of gameplay to 19,738 keyframes while keeping 98.8% of bugs. A second-stage LLM judge and image retrieval added
  little.

The hardest glitch types across studies are body poses and animation, subtle clipping, facial errors, and items that
are wrong for the context. Confidence: medium. The benchmark papers are primary, but most of the 2026 work comes from
one lab (the University of Alberta, with Sony, EA and Ubisoft) and is preprint-only. The numbers were extracted by
summarizing tools and were not checked by hand.

### 4. 2-bit quantization costs fine detail first; IQ3_XXS recovers most of it
On the ISTA-DASLab card's text benchmarks (task average), BF16 scores 93.12, IQ2_XS 89.16 (96% of BF16), IQ3_XXS
92.57 (99%) and IQ3_S 93.26. The card reports no vision results, and "RCO" means bit allocation (Riemannian
Constrained Optimization), not vision calibration. Studies of quantized vision-language models agree that text
reading (OCRBench, TextVQA) and perception (SEED) degrade first while reasoning holds up:

- MBQ, arXiv 2412.19509: with plain AWQ at 3-bit, a 72B model fell below a 7B model at FP16.
- VEQ, arXiv 2602.01037: Qwen3-VL-30B MoE scored 78.98 at BF16, 74.36 at 4-bit and 67.14 at 3-bit.
- VLMQ, arXiv 2508.03351: Qwen2.5-VL-7B collapsed at INT2.

The language model, not the vision encoder, accounts for most of the sensitivity (arXiv 2601.15287), so keeping
the mmproj at BF16 is right. These studies used uniform quantization methods on smaller models, so they overstate the
damage compared with importance-guided mixed-precision GGUF. Confidence: medium-low, because nothing measures 2-bit
on video.

## Conflicts and uncertainties
- **General video rank versus glitch detection.** Qwen3.6-27B tops the local Video-MME results, yet scores about
  52 F1 on TempGlitch, near chance. General video benchmarks do not predict glitch-detection skill.
- **Prompt sensitivity.** VideoGameQA-Bench found only a few points of variation between prompts; the 2026 clipping
  study found large precision/recall swings for open models.
- **Gemma 4 and video.** Its official card says it takes no video input, yet llama.cpp's video PR was tested on a
  Gemma 4 model by feeding it frames.
- **MiniMax M3.** Video-MME is 85.4 on llm-stats and 84.6 at 512 frames in blogs; the official card has no figure.
- **Single-source claims.** PhysGame's finding that lower resolution costs little; MiMo-V2.5's Video-MME score.

## Gaps — what we did not find
- No game-QA benchmark includes Qwen3.8-Flash-Next or Qwen3.6-35B-A3B.
- No measurement of 2-bit or 3-bit GGUF quantization on any video benchmark.
- No published accuracy comparison of llama.cpp's video path against transformers or vLLM.
- No evidence on whether thinking mode helps glitch detection.
- No study comparing frames sent as separate images against native video input.
- No per-benchmark video numbers for Molmo2; no 2026 InternVL release.

## Source quality assessment
- **Primary:** model cards (Qwen, ISTA-DASLab, Moonshot, NVIDIA, Google), source code (transformers, vLLM,
  qwen-vl-utils, llama.cpp), the Qwen3-VL tech report, the VideoGameQA-Bench and GlitchBench papers, and the
  quantization papers.
- **Preprints, not peer reviewed:** the 2026 game-QA papers (TempGlitch, RefGlitch-Bench, EA study, clipping study).
- **Tertiary, low confidence:** aggregators (benchlm.ai, llm-stats) — used for leads only; MiMo and MiniMax figures
  rest on them.

Benchmark numbers were extracted by summarizing tools and were not checked by hand against the papers' tables.

## Recommendations for this PC
1. **Make the video prompt match the reference.** Emit `<t.t seconds>` before every frame pair, pair frames as (0,1),
   (2,3), …, drop "Video:", and size frames by token budget (256–768 tokens per pair) instead of a fixed 448 px.
   This is a change to `strata-vision`.
2. **Switch to IQ3_XXS** (75.8 GB download, about 49 GB of RAM; close the browser) if fine-detail misses show up.
3. **Run the pipeline in stages:**
   - ffmpeg scene-change prefilter.
   - About 1 fps of frames checked for single-frame glitches, each next to the last clean frame, with a fixed JSON
     checklist and a tuned threshold.
   - Freezes, stuck characters, flicker and speed jumps caught by frame differencing, optical flow or engine
     telemetry.
   - Every flag goes to human review.
4. **Settings for video questions:** 2 fps (4 fps for timing-sensitive clips), short clips (10–30 s), 512–768 tokens
   per pair when on-screen text matters.
5. **Run a local bake-off** on 50–100 labelled clips from your own game, comparing Flash-Next IQ3_XXS with
   Qwen3.6-35B-A3B at Q4_K_M. No published benchmark answers which is better for your game.

## Sources
**Primary**
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- https://huggingface.co/Qwen/Qwen3.6-27B
- https://huggingface.co/moonshotai/Kimi-K2.5
- https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
- https://huggingface.co/google/gemma-4-31B-it
- https://arxiv.org/abs/2511.21631 (Qwen3-VL tech report)
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_vl
- https://github.com/QwenLM/Qwen3-VL
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py
- https://github.com/ggml-org/llama.cpp/pull/24269
- https://arxiv.org/html/2505.15952 (VideoGameQA-Bench, NeurIPS 2025)
- https://arxiv.org/html/2312.05291 (GlitchBench, CVPR 2024)
- https://arxiv.org/html/2412.01800v1 (PhysGame)
- https://arxiv.org/html/2412.19509 (MBQ)
- https://arxiv.org/html/2602.01037 (VEQ)
- https://arxiv.org/html/2508.03351v1 (VLMQ)
- https://arxiv.org/html/2601.15287v1
- https://proceedings.mlr.press/v202/dettmers23a

**Preprints**
- https://arxiv.org/html/2605.21443 (TempGlitch)
- https://arxiv.org/html/2607.25921 (clipping-detection agents)
- https://arxiv.org/html/2604.11082 (RefGlitch-Bench)
- https://arxiv.org/html/2603.22706 (EA industrial study)
- https://arxiv.org/abs/2606.17118 (MODE)
- https://arxiv.org/abs/2604.18556 (GSQ)
- https://arxiv.org/abs/2605.00649 (RCO)

**Tertiary (leads only)**
- https://benchlm.ai/benchmarks/videomme
- https://llm-stats.com/models/compare/minimax-m3-vs-qwen3.8-flash-next
