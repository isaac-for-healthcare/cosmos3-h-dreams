<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3-H-Dreams

Action-conditioned surgical video simulation with recorded action replay and
live Meta Quest controller input, built on [FlashDreams](https://github.com/NVIDIA/flashdreams).

Starting from a surgical scene image, Cosmos3-H-Dreams generates video in chunks
conditioned on instrument actions. Replay recorded actions to an MP4, or control
the simulation from Quest Browser and watch the output on the headset or a PC.

## Quick Start

Run these commands from the repository root. You need a compatible
Cosmos3-H-Dreams checkpoint and an initial scene image; model weights and scene
assets are supplied separately.

### 1. Install dependencies

With Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a compatible NVIDIA GPU environment available:

```bash
uv sync --package flashdreams-cosmos3-h-dreams --locked
```

### 2. Prepare the session

Place your checkpoint at `data_local/checkpoints/model.pt` and scene PNGs under
`data_local/scenes/`. The bundled
[real-time configuration](cosmos3_h_dreams/configs/real_time_2steps.yaml) uses the available example data in [example_data/scenes](./assets/example_data/scenes/); edit
that file if your assets live elsewhere.

For Quest Browser, generate an HTTPS certificate. Replace `<server-LAN-IP>`
with the server address reachable from the headset:

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=quest" -addext "subjectAltName=IP:<server-LAN-IP>"
```

### 3. Start the simulation

```bash
uv run --no-sync --package flashdreams-cosmos3-h-dreams \
  python -m cosmos3_h_dreams.quest_webrtc \
  --session-config cosmos3_h_dreams/configs/real_time_2steps.yaml
```

Open `https://<server-LAN-IP>:8443/quest` in Quest Browser, accept the certificate
warning, and select **Enter VR**. Wait for model loading and compilation, then
track both controllers, release the triggers, and hold still for calibration.
Open `https://<server-LAN-IP>:8443/viewer` on a PC to watch the same simulation.

The example uses two denoising steps, TAEHV decoding, compiled FadeMem caching,
and HTTPS/MJPEG video. Add `--video-transport webrtc` to use WebRTC video.
Initial compilation takes longer than subsequent generation.

For recorded actions and MP4 output, see [offline replay](cosmos3_h_dreams/README.md#offline-replay).

## Documentation

- [Setup and usage](cosmos3_h_dreams/README.md): assets, replay, Quest controls,
  session configuration, transport, and performance options.
- [Inference presets](cosmos3_h_dreams/README.md#inference-presets): decoder,
  chunk size, denoising, and cache choices.
- [Contributing](CONTRIBUTING.md): development setup and tests.
- [Development and packaging](DEV.md): versions and local distribution builds.

## System Requirements

| Component | Requirement |
| --- | --- |
| Platform | Linux with an NVIDIA CUDA-capable GPU |
| GPU memory | With the default configuration (480x832 resolution and 12 history latents), the required VRAM is over 32GB. |
| Python | 3.12 by default via `.python-version`; package range is `>=3.12,<3.15` |
| Browser test tooling | Node.js 14.18.0 or newer; use a currently supported LTS release |
| Package manager | uv 0.11.8 or newer; use the committed lockfile |
| CUDA environment | Driver compatible with the installed PyTorch CUDA build; the workspace defaults to CUDA 13 |
| Replay output | `ffmpeg` available on `PATH` for MP4 output |
| Live control | Meta Quest headset with Quest Browser and tracked controllers; HTTPS access to the server |
| Assets | Compatible checkpoint and initial frame; `[N, 28]` action array for replay |

First use may download tokenizer and decoder assets. Allow storage for these,
the model checkpoint, the Python environment, and compiler caches. GPU memory
use and throughput depend on resolution, chunk size, cache, and decoder settings.

## Resources

- [Cosmos3-H-Dreams HF](https://huggingface.co/nvidia/Cosmos3-H-Dreams): model repository for the previous version.
- [Cosmos3-H-Embodied-Nano](): Base model for Cosmos3-H-Dreams.
- [FlashDreams](https://github.com/NVIDIA/flashdreams): Underlying inference and serving framework.
- [Open-H-Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment): Multi-embodiment surgical dataset used for training.
- [NVIDIA Cosmos Platform](https://www.nvidia.com/en-us/ai/cosmos/): Product website.

## Acknowledgments

Cosmos3-H-Dreams builds on [FlashDreams](https://github.com/NVIDIA/flashdreams),
NVIDIA's inference and serving framework for autoregressive video and world
models. We thank its authors and contributors.

Our FadeMem cache mode independently implements the memory consolidation method introduced by
Yu Lu, Junjie Yang, Piotr Koniusz, YuXin Song, and Yi Yang in
[FadeMem: Distance-Aware Memory Consolidation for Autoregressive Video Diffusion](https://arxiv.org/abs/2606.10671)
(2026).

## License

Source code is distributed under [Apache-2.0](LICENSE). See
[NOTICE](NOTICE) and [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) for attribution.
Downloaded model assets retain their own license terms.
