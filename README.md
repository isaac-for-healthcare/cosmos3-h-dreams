<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3-H-Dreams

Action-conditioned surgical video simulation with recorded action replay and
live headset controller input, built on [FlashDreams](https://github.com/NVIDIA/flashdreams).

Starting from a surgical scene image, Cosmos3-H-Dreams generates video in chunks
conditioned on instrument actions. Replay recorded actions to an MP4, or control
the simulation from a headset browser and watch the output on the headset or a PC.

## Quick Start

Run these commands from the repository root. You need a compatible
Cosmos3-H-Dreams checkpoint and an initial scene image. Model weights and user
assets are supplied separately; the repository includes an
[example scene image and action trajectory](assets/example_data/scenes/), with
their provenance and licenses recorded in [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES).

### 1. Install dependencies

With Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a compatible NVIDIA GPU environment available:

```bash
uv sync --package flashdreams-cosmos3-h-dreams --locked
```

### 2. Prepare the session

Place your checkpoint at `data_local/checkpoints/model.safetensors` and scene PNGs under
`data_local/scenes/`. The bundled
[real-time configuration](cosmos3_h_dreams/configs/real_time_2steps.yaml) uses the available example data in [example_data/scenes](./assets/example_data/scenes/); edit
that file if your assets live elsewhere.

For the headset browser, generate an HTTPS certificate. Replace `<server-LAN-IP>`
with the server address reachable from the headset:

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=headset" -addext "subjectAltName=IP:<server-LAN-IP>"
```

### 3. Start the simulation

```bash
uv run --no-sync cosmos3-h-dreams-run headset \
  --session-config cosmos3_h_dreams/configs/real_time_2steps.yaml \
  --checkpoint PATH/TO/CHECKPOINT/c3-h-d_2-step_model_bf16.safetensors
```

Open `https://<server-LAN-IP>:8443/headset` in the headset browser, accept the certificate
warning, and select **Enter VR**. Wait for model loading and compilation, then
track both controllers, release the triggers, and hold still for calibration.
Open `https://<server-LAN-IP>:8443/viewer` on a PC to watch the same simulation.

The example uses two denoising steps, TAEHV decoding, compiled FadeMem caching,
and HTTPS/MJPEG video.
Initial compilation takes longer than subsequent generation.

For recorded actions and MP4 output, see [offline replay](cosmos3_h_dreams/README.md#offline-replay).

## Documentation

- [Setup and usage](cosmos3_h_dreams/README.md): assets, replay, headset controls,
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
| Live control | Headset with a WebXR-compatible browser and two tracked controllers; HTTPS access to the server |
| Assets | Compatible checkpoint and initial frame; `[N, 28]` action array for replay |

First use may download tokenizer and decoder assets. Allow storage for these,
the model checkpoint, the Python environment, and compiler caches. GPU memory
use and throughput depend on resolution, chunk size, cache, and decoder settings.

## Resources

- [Cosmos3-H-Dreams HF](https://huggingface.co/nvidia/Cosmos3-H-Dreams): model repository for the previous version.
- [Cosmos3-H-Embodied-Nano](https://github.com/NVIDIA-Medtech/Cosmos3-H-Embodied-Nano): Base model for Cosmos3-H-Dreams.
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

Source code is distributed under [Apache-2.0](LICENSE). The
`cosmos3_h_dreams/` package is under the same Apache License 2.0 and the same
NVIDIA copyright; its copy is included for the package build at
[cosmos3_h_dreams/LICENSE](cosmos3_h_dreams/LICENSE). See
[NOTICE](NOTICE) and [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) for attribution.
Downloaded model assets retain their own license terms.

## Automatically retrieved materials

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.

Depending on the selected CUDA profile, optional extras, decoder and checkpoint
location, installation or runtime loading may retrieve the following NVIDIA
materials. The selected package versions are recorded in `uv.lock`; each
material retains its own applicable terms.

| NVIDIA materials | Available from |
| --- | --- |
| CUDA Toolkit metapackages and CUDA libraries (runtime, cuBLAS, cuFFT, cuFile, cuRAND, cuSOLVER, cuSPARSE, cuSPARSELt, CUPTI, NVRTC and nvJitLink) | [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), [cuda-toolkit on PyPI](https://pypi.org/project/cuda-toolkit/) |
| CUDA Python bindings and path discovery | [cuda-bindings](https://pypi.org/project/cuda-bindings/), [cuda-pathfinder](https://pypi.org/project/cuda-pathfinder/) |
| cuDNN | [CUDA 12 package](https://pypi.org/project/nvidia-cudnn-cu12/), [CUDA 13 package](https://pypi.org/project/nvidia-cudnn-cu13/) |
| NCCL | [CUDA 12 package](https://pypi.org/project/nvidia-nccl-cu12/), [CUDA 13 package](https://pypi.org/project/nvidia-nccl-cu13/) |
| NVSHMEM | [CUDA 12 package](https://pypi.org/project/nvidia-nvshmem-cu12/), [CUDA 13 package](https://pypi.org/project/nvidia-nvshmem-cu13/) |
| NVTX and NVIDIA management-library Python bindings | [nvidia-nvtx](https://pypi.org/project/nvidia-nvtx/), [nvidia-nvtx-cu12](https://pypi.org/project/nvidia-nvtx-cu12/), [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) |
| Transformer Engine and its CUDA/PyTorch components (development extra) | [Transformer Engine](https://github.com/NVIDIA/TransformerEngine), [transformer-engine on PyPI](https://pypi.org/project/transformer-engine/) |
| NVIDIA VFX bindings (optional RTX post-processing extra) | [nvidia-vfx](https://pypi.org/project/nvidia-vfx/) |
| NVIDIA model checkpoints, when the user supplies a remote checkpoint location | The user-selected source; [Cosmos3-H-Dreams model repository](https://huggingface.co/nvidia/Cosmos3-H-Dreams) hosts the previously released version |

Third-party model and tokenizer materials may also be retrieved on first use:

- [Wan2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/Wan2.2_VAE.pth).
- [TAEHV decoder](https://github.com/madebyollin/taehv), including the
  [taew2_2.pth mirror](https://huggingface.co/lightx2v/Autoencoders/blob/main/taew2_2.pth).
- [Qwen3-VL-8B-Instruct tokenizer](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

Model weights and tokenizer assets are supplied separately from this source
release. See [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) for dependency attribution
and [NOTICE](NOTICE) for the separately licensed example image included here.
