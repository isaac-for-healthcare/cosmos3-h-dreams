<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Try the eight-frame latency improvements

This evaluation branch targets lower control-to-browser latency and steadier
presentation while retaining **eight output frames per model step**. It builds
on [Nishant Jain’s keyboard/token fork](https://github.com/nisjain-123/cosmos3-h-dreams/tree/f7b2821f027612d0e39c34db3fd0635bfab9a262).
The native headset/WebXR entry point remains available.

Status: **useful opt-in on the tested H100/Chromium stack**. RTX PRO 6000 is a
validation target, not a GPU on which these numbers have been reproduced.
No model weights, attention/cache semantics, or shared `flashdreams/` code were
changed by our optimization patch. The inherited token fork has its own
[provenance](../TOKEN_DEMO.md#provenance).

## Quick start: keyboard, no headset required

Use a Linux CUDA machine and follow the [system requirements](../README.md#system-requirements).
Use a driver/PyTorch build supporting the actual RTX PRO 6000 variant. The locked
workspace defaults to CUDA 13. Supply the public two-step BF16 checkpoint from
[the model repository](https://huggingface.co/nvidia/Cosmos3-H-Dreams) separately.
The source fork does not redistribute downloaded weights or decoder exports.

```bash
git clone --branch codex/steady8-keyboard --single-branch \
  https://github.com/huvers/cosmos3-h-dreams.git
cd cosmos3-h-dreams
uv sync --package flashdreams-cosmos3-h-dreams --extra token-stream --locked

.venv/bin/python -m cosmos3_h_dreams.token_demo \
  --host 127.0.0.1 --port 8081 \
  --checkpoint /path/to/c3-h-d_2-step_model_bf16.safetensors \
  --initial-frame assets/example_data/scenes/example_frame.png \
  --config wan-taehv-chunk2-2steps-fademem-compile \
  --display-fps 30 \
  --deadline-pacing \
  --gpu-frame-conversion \
  --dedicated-inference-thread
```

The model preset keeps `chunk_size=2` (two latents/eight output frames), seed 0,
832×480 resolution, two denoising steps, TAEHV, compiled denoising/CUDA graphs,
and cosine FadeMem with 12 slots/beta 0.3. Model conditioning remains **10 Hz**;
`--display-fps 30` changes presentation only. Do not substitute `--fps 30`.
The startup log prints the active model/presentation settings. Wait for model
loading, compilation and decoder export to complete before connecting.

On the same workstation, open <http://127.0.0.1:8081/keyboard>. From a separate
client, keep this SSH forward open and use <http://127.0.0.1:18081/keyboard>:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18081:127.0.0.1:8081 user@server
```

Select **Pixel · MJPEG**, then **Connect / restart**. The server deliberately
binds to loopback; remote clients use SSH. Only one generation session can run
at a time. First setup may download decoder assets and export ONNX. Raw-token
mode also requires Chrome/Edge WebGPU and access to the pinned ORT Web CDN.

- Left tool: **W/A/S/D**, **Q/E** for depth, **Z** for jaw closure.
- Right tool: **arrows**, **Page Up/Down** for depth, **X** for jaw closure.
- Release keys to hold position; keep the page focused.
- Try a movement, release, reversal, and jaw toggle after warmup.
- **Stop**, then reconnect to reset to the initial scene.

The fork’s keyboard adapter is experimental. A visually plausible response is
not evidence of calibrated instrument control. Use the existing headset entry
point when testing the original controller input path.

## Headset entry point

With checkpoint and HTTPS certificate/key prepared as described in the
[upstream quick start](../README.md#quick-start):

```bash
uv run --no-sync cosmos3-h-dreams-run headset \
  --session-config cosmos3_h_dreams/configs/steady8_2steps.yaml \
  --checkpoint /path/to/c3-h-d_2-step_model_bf16.safetensors
```

This preset keeps eight-frame model generation, 30 Hz headset sampling and
30 fps presentation. Keyboard latency and headset latency have different input
acquisition behavior and must be reported separately.

## What changed

1. Convert generated RGB frames to uint8 on the GPU before host transfer.
2. Pace MJPEG publication against a deadline that includes JPEG encoding time.
3. Send complete next-part MJPEG headers immediately so Chromium can display the
   current image without waiting for the next generated image.
4. Keep initialization, inference, GPU preparation and cleanup on one dedicated
   worker to avoid the worker-switch stall observed in the tested runtime.

The first, second and fourth changes remain opt-in. Their inverse flags are
`--no-gpu-frame-conversion`, `--no-deadline-pacing` and
`--no-dedicated-inference-thread`. Turning them off does **not** undo the MJPEG
header fix. Defaults otherwise retain the fork’s original presentation behavior.
The documented command enables the combination exercised by the H100 harness.

## Evidence and limits

These are short software-timing experiments, not optical display/physical
headset measurements. Results do not guarantee the same speedup on every GPU,
client, network or operating system.

### Original eight-frame pipeline comparison

H100, same eight-frame grouping, 30 fps presentation, GPU conversion and deadline
pacing in both rows; measured before versus after eager headers plus the dedicated
worker. Stage profiling was disabled during timing.

| Pipeline | Median input-to-browser | P95 | Largest observed frame gap |
| --- | ---: | ---: | ---: |
| Before header/worker fixes | 604.9 ms | 638.4 ms | 733.5 ms |
| Combined fixes | 495.9 ms | 544.8 ms | 50.7 ms |

The combined result saved 109 ms / 18.0% in that Eos-local browser trial.
512 replay frames matched exactly across cache boundaries; both resets reproduced
the initial eight frames. This does not establish identical pixels between GPU
architectures. Earlier four-frame generation showed longer-rollout drift, so this
trial keeps eight frames.

### Remote keyboard transport comparison

Same H100 model settings, actual Chromium 156 on an Apple/Metal client through
SSH/VPN, Pixel → Raw → Raw → Pixel order. Four 45-second control sequences;
exclude the first 10 seconds and final 3 seconds of each sequence. Each transport
has 64 measured state changes across two 32-second windows, all consumed and
observed. The MJPEG endpoint is decoded-image availability sampled on animation
frames; the token endpoint is canvas submission. Neither is physical scanout or
semantic motion-onset detection.

| Transport | Median | P95 | Observed FPS | Gaps over 100 ms |
| --- | ---: | ---: | ---: | ---: |
| Optimized MJPEG | 624.4 ms | 783.0 ms | 29.1–29.2 | 3 |
| Raw FP16 latents | 749.4 ms | 937.5 ms | 18.4–19.2 | 171 |

MJPEG had 25 intermediate generated frame IDs not observed between window
endpoints; raw presentation had none, but more stalls. Thus the smoother result
is not a claim of identical frame delivery. Pretrial HTTP echo medians ranged
138–171 ms. The native path buffers a playing and a prepared block; raw uses a
two-latent window acknowledged on presentation. Queue policies differ.

Browser FP32 TAEHV with a GPU-resident cache took 146.3 ms median per four frames,
above the 133.3 ms budget for 30 fps. A 32-frame same-latent comparison against
native H100 decoding passed: MAE 0.603/255 and PSNR 49.59 dB, against predetermined
limits of MAE ≤3 and PSNR ≥32 dB. Longer latent-rollout quality remains unvalidated.
If the browser decoder runs on RTX instead of Apple hardware, repeat this test;
the Apple result does not establish which transport wins there.

### Validation scope of this public branch

The merged server implementation passed the live H100/browser experiment. This
publication adds CLI wiring for the same server options and a headset YAML
preset; 74 CPU tests and 6 CUDA conversion tests passed on H100, along with the
local browser JavaScript checks. The CLI wiring was tested without loading a
model; the new launcher command has not had a separate model-inference run.
No RTX PRO 6000 measurements have been made. Relevant source base identities
are the original release
`fb18792d8af680a713801dea7688919e46329810` and token fork
`f7b2821f027612d0e39c34db3fd0635bfab9a262`.

## What to report from RTX PRO 6000

Record the exact GPU edition, driver/PyTorch/CUDA versions, git commit, browser,
client GPU, transport, and whether the client is local or remote. After warmup,
compare with the existing deployment at the **same eight-frame model settings**
and presentation rate. Describe response to direction reversals, visible pauses,
and whether Stop/reconnect works. Report cold startup separately.

For quantitative acceptance, repeat the same command schedule and frame-ID
instrumentation, measure p50/p95/p99 command-to-first-corresponding-frame and
frame gaps, and verify replay/reset quality. This quick-start serves the normal
demo; it does not claim to produce automatic latency measurements. Do not call
an opt-out run a complete baseline because it retains the header fix.

## CPU checks

With the repository’s test dependencies installed:

```bash
PYTHONPATH=cosmos3_h_dreams:flashdreams .venv/bin/python -m pytest \
  cosmos3_h_dreams/tests/test_token_demo_options.py \
  cosmos3_h_dreams/tests/test_headset_server.py \
  cosmos3_h_dreams/tests/test_session_config.py \
  cosmos3_h_dreams/tests/test_token_stream.py -m ci_cpu -q
node cosmos3_h_dreams/tests/test_headset_status.cjs
node cosmos3_h_dreams/tests/test_scene_selector.cjs
node --test cosmos3_h_dreams/tests/test_token_playout.mjs
```
