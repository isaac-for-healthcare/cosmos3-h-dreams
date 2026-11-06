# Cosmos3 keyboard and token-streaming demo

This is an experimental port, not an upstream Cosmos3 feature. It generates
new frames with the actual Cosmos3 checkpoint; it does not replay a video.
Keyboard motion is a new adapter to Cosmos3's instrument-action contract,
not physical robot control or a clinically validated controller.

## Provenance

- Cosmos3 base: `fb18792d8af680a713801dea7688919e46329810`.
- FlashDreams transport/export/browser source: branch
  `dev/ksheth/video_token_streaming_v2`, commit
  `ab1c5698daf098bdb7111a6d0d11012151a64dc9`.
- Shared token framing, raw-FP16 codec, and emitter were imported unchanged.
- Adaptations: Wan2.2 48-channel TAEHV decoder, initial-reference cache seed,
  keyboard input, bounded presentation acknowledgments, and session cleanup.
- ONNX Runtime Web is pinned to 1.30.0. In the Windows Edge A/B test, 1.20.1
  produced flat output; 1.30.0 produced changing surgical frames with the
  same model. This is a compatibility observation, not a kernel root cause.

For this fork’s optimized eight-frame, compiled 30 fps evaluation, use the
[latency trial guide](cosmos3_h_dreams/LATENCY_TRIAL.md). The original eager
demo command below remains available as a separate configuration.

## Setup and launch

Run from the repository root on a supported Linux NVIDIA GPU machine. Follow
[the upstream requirements](README.md#system-requirements) and obtain the
Cosmos3 checkpoint under its applicable model terms. Install the ONNX export
extra and use the repository's locked environment:

```bash
uv sync --package flashdreams-cosmos3-h-dreams --extra token-stream --locked
.venv/bin/python -m cosmos3_h_dreams.token_demo \
  --host 127.0.0.1 --port 8081 \
  --checkpoint /path/to/c3-h-d_2-step_model_bf16.safetensors \
  --initial-frame assets/example_data/scenes/example_frame.png \
  --config wan-taehv-chunk2-2steps-fademem-compile \
  --no-compile-denoise --no-cuda-graph-denoise \
  --eager-warmup-steps 8
```

The optional eager warmup moves first-use kernel initialization before readiness
and resets the rollout afterwards. Omit that flag to retain the zero-warmup
default. The eager command above disables compilation/CUDA graphs; the
latency trial guide instead uses the compiled preset with one dedicated
inference worker, matching the configuration exercised on H100.
Weights are supplied separately; first launch may download the TAEHV weights
and export the FP32 ONNX decoder under `data_local/decoder/`. Do not share model
weights, generated decoder files, or local caches as part of the source patch.

Keep the server running in a persistent terminal. Check `/healthz` for `ready`
before connecting; HTTP 200 for the page alone does not prove model readiness.

From the client computer, replace `user@server` with your SSH login and keep
this terminal open:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18081:127.0.0.1:8081 user@server
```

Open `http://127.0.0.1:18081/keyboard` in a WebGPU-capable Chrome or Edge.
Localhost provides the secure browser context needed for WebGPU. Closing the
SSH forwarding terminal makes this address unavailable. Only one generation
session is accepted at a time; press Stop before handing the demo to someone
else. Closing the browser also releases the session.

Select Pixel or Raw Tokens, then Connect:

- Left instrument: W/A/S/D, Q/E depth, Z closes the jaw.
- Right instrument: arrow keys, Page Up/Down depth, X closes the jaw.
- Release keys to stop changing position; reconnect resets the scene.
- Browser focus is required. Lost input expires after one second.
- Completed short presses between generation steps are retained for one action
  row. Repeated pending presses coalesce instead of accumulating a queue.

The server serves the decoder artifact. Clients do not install ONNX or download
weights manually. Browser access to jsDelivr is required for pinned ONNX Runtime
Web 1.30.0 and its WASM assets. VR hardware is not needed for keyboard mode.
Only the bundled default scene is configured. For recorded actions, select
`--controls replay` and supply `--actions /path/to/trajectory.npy`.

## Data paths

```text
Keyboard -> JSON keys over WebSocket -> bounded 28D poses -> padded 64D actions
Initial RGB -> native encoder -> reference latent -> Cosmos3 generation
                                                    |
                      +-----------------------------+-------------------+
                      |                                                 |
Pixel: server TAEHV -> RGB -> JPEG -> HTTP MJPEG -> browser image         |
Raw: cropped BF16 latent -> FP16 bytes -> binary WebSocket <--------------+
     -> FP32 browser tensor -> ONNX TAEHV/WebGPU -> RGB canvas
```

This Cosmos3 app uses **MJPEG**, not H.264/WebRTC/NVENC, for pixel delivery.
Raw delivery bypasses generated-frame server decode and JPEG encoding. Model
initialization still primes the native decoder. Both paths use the same
generation model; browser TAEHV has its own persistent temporal cache.

At the shipped 832x480 configuration: one latent is 48x30x52, 149,760 FP16
payload bytes plus framing, and decodes to four RGB frames. Spatial reduction
is 16x16 and temporal reduction is 4. This is continuous-valued latent
transport, not integer vocabulary IDs or an entropy-compressed bitstream.
The reference seed initializes the cache and is not presented as generated
video. Two unacknowledged latent chunks bound transport buffering.
Raw presentation starts with the first decoded frame (no eight-frame preroll).
Every latent still passes through the causal decoder; latents are not dropped.
Pixel playback already retains only the latest JPEG for slow viewers.

## Validation and limitations

Run these CPU checks in an environment with the product and test dependencies
installed. They do not load a checkpoint or exercise GPU generation:

```bash
PYTHONPATH=cosmos3_h_dreams:flashdreams .venv/bin/python -m pytest \
  cosmos3_h_dreams/tests \
  flashdreams/tests/test_token_stream_codec.py \
  flashdreams/tests/test_token_stream_emitter.py \
  flashdreams/tests/test_token_stream_framing.py \
  tests/test_dependency_policy.py tests/test_agent_skills.py -m ci_cpu -q
node --test cosmos3_h_dreams/tests/test_token_playout.mjs
node cosmos3_h_dreams/tests/test_headset_status.cjs
node cosmos3_h_dreams/tests/test_scene_selector.cjs
```

Tests cover latent output without generated-frame server decode, codec/framing,
flow control and disconnect cleanup, keyboard release/timeout, short presses,
and presentation pacing. GPU/browser acceptance is separate: verify changing
frames and controls in both modes, Stop/reconnect, and browser-close cleanup.
A passing CPU suite does not establish ONNX/WebGPU parity or headset operation.

Keyboard targets are bounded but require further calibration/quality assessment
for the checkpoint. The configured 10 fps is playout cadence, not a throughput
claim. The service has no application authentication and deliberately binds to
loopback; use authenticated TLS deployment before exposing it publicly.
