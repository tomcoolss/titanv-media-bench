# Titan V (12 GB) as a media side-card — vision + ASR + img2img

Real-world benchmark of a first-gen Volta **TITAN V (GV100, 12 GB HBM2)** running as a
multimodal side-card next to an always-on LLM brain on separate cards.
All numbers measured 2026-10-22 on a live agent host (the text LLM on the other GPUs
served traffic throughout; nothing here required stopping it).

## Hardware / software

| item | value |
|---|---|
| GPU | NVIDIA TITAN V 12 GB (Volta sm_70), **power-capped 120 W** (stock 250 W) |
| Slot | PCIe Gen3 x8 (chipset branch), riser cable |
| Driver | 580.178.04 (last branch supporting Volta) |
| Toolkit | CUDA 12.8 (relocatable), gcc 13 host compiler |
| OS | Ubuntu, kernel 7.0 |
| Vision engine | llama.cpp `d772569` (llama-server 0.7.0-dev), CUDA sm_70, `-ngl 99`, flash-attn, ctx 16K |
| Vision model | `Qwen3-VL-8B-Instruct` UD-Q4_K_XL + `mmproj-F16` (unsloth GGUF) |
| ASR engine | whisper.cpp `231638a` (1.9.5), CUDA sm_70, OpenAI-compatible `/v1/audio/transcriptions` |
| ASR model | `ggml-large-v3-turbo.bin` |
| img2img | Z-Image-Turbo via diffusers 0.36 / torch 2.5.1+cu121, fp32 + `enable_sequential_cpu_offload()`, `cudnn.enabled=False` (Volta VAE conv needs it), VAE decode fp16 |

VRAM budget: llama-server Q4_K_XL + KV ≈ 9.6 GB, whisper turbo ≈ 2.1 GB → both engines
resident side-by-side in 12 GB. Z-Image time-shares the card: the two servers stop for the
duration of an edit (fp32 sequential offload streams 25 GB of weights from host RAM and needs
the card to itself), then resume automatically via systemd (`Restart=always` + trap).

## Results

### Vision — Qwen3-VL-8B on llama.cpp (256-token answers, temp 0.1)

| metric | cold (first request per image) | warm (KV prefix cache hit) |
|---|---|---|
| wall, 4.1K-token image prompt | 15.5 s | 5.9 s |
| wall, 4.0K-token image prompt | 14.8 s | 4.6 s |
| wall, 3.4K-token image prompt | 12.5 s | 4.6 s |
| decode (wall-derived, incl. mmproj re-encode) | — | 49–58 tok/s |

Text-only baseline (no image): **68.8 tok/s** decode median, 3 runs.
Cold = mmproj vision-tower encode + full prefill; warm = identical image+prompt repeated
(prefix cache), mmproj still re-runs per request. Qualitative: dense OCR (math on a
blackboard, mixed CJK posters) came back fully correct, zh-Hans/zh-Hant both handled.

### ASR — whisper large-v3-turbo (3 clips × 3 runs: EN, zh-CN, zh-HK TTS)

| metric | value |
|---|---|
| RTF median | **0.031** (≈ 32× real-time) |
| 11 s clip | 0.30–0.38 s wall |

### img2img — Z-Image-Turbo product-photo edit (fp32, streaming offload)

| metric | value |
|---|---|
| 992×992, strength 0.4, 9 steps (4 DiT forwards) | **80 s total** = 16 s model load (NVMe) + 63 s sample + VAE |

For reference, the same fp32 recipe on a Tesla V100-PCIE-16GB (uncapped) measured 82 s at
1024²/9 steps — the 120 W-capped Volta clone is at parity within noise, and the point of the
side-card is that **no LLM downtime** is incurred at all.

## Notes for anyone copying this setup

- 120 W cap: peak observed during vision load ≈ cap; decode speed already reported above.
- Volta has no bf16/fp8 — fp32 is the only correct dtype for Z-Image (fp16 → NaN latents).
- ggml builds must disable OpenMP when cross-linking with a relocated gcc toolchain
  (static `libgomp.a` cannot link into a shared object); GPU inference does not miss it.
- whisper.cpp server path remap `--request-path /v1 --inference-path /audio/transcriptions`
  makes it drop-in for any OpenAI SDK `audio.transcriptions.create` caller.

Raw per-run data: [`results.json`](results.json).
