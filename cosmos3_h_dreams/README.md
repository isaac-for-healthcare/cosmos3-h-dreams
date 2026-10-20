<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3-H-Dreams setup and usage

Cosmos3-H-Dreams generates surgical video from an initial frame and instrument
actions. It supports offline replay and live headset browser sessions using
the included FlashDreams framework. Start with the repository
[Quick Start](../README.md#quick-start) for a single launch command.

All commands below run from the repository root. The distribution name is
`flashdreams-cosmos3-h-dreams`, the Python import is `cosmos3_h_dreams`, and the
application name is `cosmos3-h-dreams`.

## Installation and assets

```bash
uv sync --package flashdreams-cosmos3-h-dreams --locked
```

Install system `ffmpeg` for MP4 output. See the root
[System Requirements](../README.md#system-requirements) for the GPU environment.
The commands use `--no-sync` after installation to keep the existing environment.

Supply a checkpoint and matching scene assets. For reference, example data is available in [example_data/scenes](../assets/example_data/scenes/).
The bundled image comes from `episode_004148` of the
CMR surgical (cholecystectomy) data in
[Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
and retains its [CC-BY-4.0 license](../LICENSES/CC-BY-4.0.txt). See the
[example-data attribution](../THIRD-PARTY-NOTICES)
for contributor credit and format information.
The bundled `trajectory.npy` is an NVIDIA-generated example action trajectory
for this project, licensed under Apache-2.0. It is
separate from the dataset-derived image.

Asset paths resolve from the launch directory, including
when a session YAML lives in another directory. Supply the checkpoint through
`--checkpoint` or `runtime.checkpoint` in the session YAML; the CLI option
overrides the YAML value.

Replay actions must be a finite `float32` NumPy array shaped `[N, 28]`. Four action
rows correspond to one latent video frame. Use actions normalized for the
checkpoint, or supply `--normalizer /path/to/normalizer.npz` with 28-element
`offset` and `scale` arrays. JSON normalizers are not accepted. Do not normalize
an already-normalized replay twice. The runtime pads actions to the model's
64-dimensional input width.

The Wan VAE, tokenizer, and standard TAEHV decoder can fetch assets on first use.
The optional `taehv2super` decoder requires an accessible checkpoint specified by
`COSMOS3_H_DREAMS_TAEHV2SUPER_CHECKPOINT`.

## Inference presets

Select a model preset with `--config`. Presets supply inference settings; asset
paths and output destinations are configured separately. Explicit CLI options
override preset values. Definitions live in
[config.py](cosmos3_h_dreams/config.py).

| Preset | Decoder | Latent frames/chunk | Denoising steps | Cache |
| --- | --- | --- | --- | --- |
| `base` | Wan | 4 | 4 | Eager `cat` |
| `wan-wan-chunk4-block-compile` | Wan | 4 | 4 | Compiled block |
| `wan-taehv-chunk4-block-compile` | TAEHV | 4 | 4 | Compiled block |
| `wan-wan-chunk3-block-compile` | Wan | 3 | 4 | Compiled block |
| `wan-taehv-chunk3-block-compile` | TAEHV | 3 | 4 | Compiled block |
| `wan-wan-chunk2-block-compile` | Wan | 2 | 4 | Compiled block |
| `wan-taehv-chunk2-block-compile` | TAEHV | 2 | 4 | Compiled block |
| `wan-taehv-chunk2-fademem-compile` | TAEHV | 2 | 4 | Compiled FadeMem |
| `wan-taehv-chunk2-2steps-fademem-compile` | TAEHV | 2 | 2 | Compiled FadeMem |

All named presets use the Wan image encoder, a 480 × 832 canvas, domain 25,
and 10 Hz model timing. Compiled presets enable CUDA graphs and overlapping
cache updates. FadeMem presets select cosine merge ranking. Choose settings
that match the checkpoint's training conditions; other resolutions and domain
IDs require separate validation. Height and width must be divisible by 16.

The bundled [real_time_2steps.yaml](configs/real_time_2steps.yaml) selects the
last model preset and the independent `headset-base` transport preset. `headset-base`
uses controller acquisition at up to 30 Hz and MJPEG playback at 20 fps. Without
that headset preset, the server defaults to MJPEG at 30 display fps and acquisition
at the model rate. Changing display FPS only changes playback pacing; changing
model FPS changes temporal conditioning.

We recommend using the 4-step checkpoint (`c3-h-d_4-step_model_bf16.safetensors`) with the `wan-taehv-chunk2-fademem-compile` preset, and the 2-step checkpoint (`c3-h-d_2-step_model_bf16.safetensors`) with the `wan-taehv-chunk2-2steps-fademem-compile` preset. Mixing them may result in suboptimal visual quality.

## Offline replay

```bash
uv run --no-sync --package flashdreams-cosmos3-h-dreams \
  flashdreams-run cosmos3-h-dreams \
  --output mp4 --output-path replay.mp4 \
  --config wan-taehv-chunk2-block-compile \
  --checkpoint checkpoints/model.safetensors \
  --initial-frame assets/example_data/scenes/example_frame.png \
  --actions assets/example_data/scenes/trajectory.npy
```

Block cache and FadeMem require full chunks, including the final chunk: the action row
count must be divisible by `4 × chunk_size` (8 for chunk2, 12 for chunk3,
16 for chunk4). Use eager `cat` for a partial final chunk, for example with
`--config base`.

For available options:

```bash
uv run --no-sync --package flashdreams-cosmos3-h-dreams \
  flashdreams-run cosmos3-h-dreams --help
```

## Headset browser sessions

Generate `cert.pem` and `key.pem` as shown in the root
[Quick Start](../README.md#quick-start). Keep the key private. The certificate
must cover the IP or hostname used by the headset; regenerate it if that changes.

```bash
uv run --no-sync cosmos3-h-dreams-run headset \
  --session-config cosmos3_h_dreams/configs/real_time_2steps.yaml
```

The launcher lists the available application names when a name is unknown.
Use `cosmos3-h-dreams-run --help` for launcher help, or
`cosmos3-h-dreams-run headset --help` for headset options. After updating
an existing checkout, run the installation command again to install the launcher
before using `uv run --no-sync`.

The server binds to `0.0.0.0:8443` by default. Open
`https://<server-LAN-IP>:8443/headset` in the headset browser and select **Enter VR**.
The initial image is displayed during startup. Compiled configurations run
warmup before accepting live input; the page reports loading, compilation, and
readiness. The first startup with a cold compiler cache can take time.

Track both controllers, release both triggers, and hold still for calibration.
A stable pose is captured after 0.3 seconds within the configured movement
thresholds. One headset controls a session. A disconnect clears calibration;
reconnecting requires calibration again.

Open `https://<server-LAN-IP>:8443/viewer` on a PC to watch without taking control.
One PC viewer is supported. Both pages provide a scene gallery when multiple
initial frames are configured. Switching scenes keeps weights loaded and resets
generation history and calibration.

### Session YAML

```yaml
runtime:
  preset: wan-taehv-chunk2-2steps-fademem-compile
  checkpoint: data_local/checkpoints/model.safetensors

headset:
  preset: headset-base
  cert: cert.pem
  key: key.pem

scenes:
  - initial_frame: data_local/scenes/*.png
```

`runtime.preset` selects model settings and `headset.preset` selects transport/input
settings. Other options use Python field names with underscores. CLI options
override YAML; `--config` and `--headset-config` replace the corresponding preset
selection. Scene CLI overrides such as `--prompt` apply to every scene.

A single-image scene requires a unique `name` and `initial_frame`; its `label`
defaults to the name. Wildcards expand in sorted path order and use each filename
as its name and label. Empty matches and duplicate names fail validation.
Optional scene fields include `prompt`, `domain_id`, `normalizer`, and replay
`actions`; the headset server uses live controller input. Omitted scene options
inherit preset/default values. Checkpoint, timing, resolution, and cache settings
are shared across scenes. All relative paths, including certificate paths,
resolve from the launch directory.

### Controls

- Controller translation and rotation drive the corresponding instrument from
  the calibrated reference pose.
- Index triggers close the jaws when squeezed and reopen them when released.
  `--headset-jaw-open` and `--headset-jaw-closed` set model-space targets (defaults
  `+1` and `-2`).
- Grip buttons drive the energy channel. Each controller’s primary button supplies the primary-button channel.
- Hold both controllers’ primary buttons for one second to restart and recalibrate. The page also provides
  **Reset simulation**.

Left-controller action channels are translation `0:3`, rotation `3:9`, trigger
`9`, grip `10`, primary button `11`, and stick `12:14`. Right-controller channels
are translation `14:17`, rotation `17:23`, trigger `23`, grip `24`, primary button
`25`, and stick `26:28`.

### Video transport and remote access

Video uses MJPEG over HTTPS. Controller poses arrive over a same-origin
WebSocket.

MJPEG can carry video and controls through a single TCP tunnel. From a PC that
can reach a remote server:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L <PC-LAN-IP>:8443:127.0.0.1:8443 <user>@<server>
```

Allow inbound TCP 8443 on that PC, use a certificate covering `<PC-LAN-IP>`, and
open `https://<PC-LAN-IP>:8443/headset` on the headset. Slow viewers skip frames
rather than accumulating an unbounded queue.

## Performance and diagnostics

Append `--profile` to record stage timings, or `--print-actions` to the headset
command to log normalized action rows and sampled controller poses. Action rows
are consumed and printed in chunks. Profiling synchronizes GPU work and can
change performance; compare steady-state runs after compilation and cache fill,
then repeat without profiling.

On CUDA, `--profile` also reports peak and time-weighted average VRAM allocated
and reserved by PyTorch, including weights, KV cache, and decoder memory. Peaks
use allocator counters; averages sample every 10 ms and can miss shorter spikes.
The measurement runs from the first inference chunk through session close,
including time between chunks, and starts fresh after a reset or scene switch.
Loading and prewarm are excluded. CUDA context memory, allocations outside
PyTorch, and other processes are not included; reserved memory includes unused
space held by PyTorch's caching allocator.

Model time, acquisition time, and display time are separate. A chunk2 step uses
eight actions. Acquiring measured poses at 30 Hz retimes those actions into the
10 Hz model timeline; it does not establish physical motion parity. Increasing
display FPS does not increase generation throughput or interpolate frames.

Playback overlap is enabled by default. `--no-overlap-playback` selects sequential
generation and playback. For MJPEG diagnostics, compare `input`, `generation`,
`queue_wait`, and `boundary_interval` after warmup. `pose_age_first_frame` runs
from server receipt of the newest pose in a block to its first JPEG publication;
it excludes network delivery and headset rendering.

### Cache options

- `cat`: eager rolling cache; accepts partial final chunks.
- `block`: fixed-buffer cache; requires full chunks. `--window-frames` controls
  the rolling attention history.
- `fademem`: bounded historical entries, configured with `--fademem-slots` and
  `--fademem-beta`. `--window-frames` does not control this mode.

The FadeMem cache is an independent implementation of the method from
[FadeMem: Distance-Aware Memory Consolidation for Autoregressive Video Diffusion](https://arxiv.org/abs/2606.10671)
by Yu Lu et al. (2026) to Cosmos3-H-Dreams.

FadeMem merges adjacent historical entries when its budget is exceeded and
protects the newest entry. Choose `--fademem-merge-selection age` or `cosine`;
cosine ranks clean latent frames and can miss small tool motion against a large
stationary background. Cache consolidation can lose scene detail. Validate long
rollouts and reversals on your checkpoint before choosing it for a workload.

Compiled presets already enable `--compile-denoise`, `--cuda-graph-denoise`, and
`--overlap-cache-update`. For comparisons, hold the checkpoint, initial frame,
actions, seed, decoder, and attention budget fixed. CPU checks cover cache logic;
they do not establish GPU speedups or visual quality.

## Tests

From an environment with the test dependencies installed and Node.js 14.18.0
or newer for the browser JavaScript checks (a currently supported LTS release
is recommended):

```bash
PYTHONPATH=cosmos3_h_dreams:flashdreams .venv/bin/python -m pytest \
  cosmos3_h_dreams/tests -m ci_cpu -q
node cosmos3_h_dreams/tests/test_headset_status.cjs
node cosmos3_h_dreams/tests/test_scene_selector.cjs
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for test dependency installation and
repository checks. CPU/browser checks use fake model sessions and do not run
GPU inference or validate live headset behavior.
