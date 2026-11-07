# Experimental FP8 generator MLPs for RTX PRO 6000 Blackwell

Status: **Deferred pending CUDA, target-GPU speed and visual-quality validation.**
Default: **off**. No model speedup or control-to-screen reduction is claimed yet.
This branch adds a separate FP8 weight overlay and runtime path; it does not change
any published checkpoint or the shared `flashdreams/` implementation.

## Decision

Start with FP8 E4M3 weights and dynamic per-token FP8 activations in the generator
MLPs. This targets dense matrix multiplication in both denoising and clean-cache
refresh. All 36 generator MLPs (108 projections) are eligible. Gate and up share
one activation quantization; down quantizes its own input. Scaling is per output
channel for weights and per row for activations. GEMM outputs and intervening
SwiGLU math stay BF16; the existing GEMM uses its default full accumulation.
Attention, action conditioning, understanding weights, norms, video readout and
VAE retain their original precision. No steps, frames or cache entries are removed.

The repository already provides `QuantizedNonPersistentLinear`, a Triton
quantizer and PyTorch `_scaled_mm`. Reuse those instead of introducing TorchAO or
Transformer Engine. PyTorch's
[2.11 rowwise implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/RowwiseScaledMM.cu)
contains SM120 kernels. Target PyTorch 2.11+ with CUDA 12.8+ (prefer the existing
CUDA 13 stack); actual support is checked with both matrix orientations before
loading the model. Hardware capability alone is insufficient. An unsupported
kernel fails clearly; there is no silent BF16 fallback.

FP8 precedes NVFP4 because this pilot prioritizes visual fidelity, has existing
kernels, and can be checked on H100 before the RTX trial. FP4 needs its own kernel
and calibration/quality investigation. Arbitrarily zeroing dense weights does not
reduce the dense matrix operations. Layer removal or fewer denoising steps changes
the model and needs retraining/distillation evidence. FP8 weight-only storage would
not necessarily accelerate these large matrix operations.

At the released 4096/12288/36 geometry, eligible BF16 matrices occupy 10.87 GB;
FP8 values plus scales occupy about 5.44 GB. The predicted resident-weight saving
is about 5.43 GB (5.06 GiB), **not half of total VRAM**. Loading currently stages the
full BF16 model plus the FP8 overlay before replacement, so startup peak memory
can increase. Actual allocated/reserved memory is measured separately.

## Separate weights and rollback

Use a new output directory. The exporter refuses to overwrite one. This pilot
accepts a local single-file BF16 safetensors checkpoint, using the same exported
key mapping as the model loader. It writes 36 independent FP8 safetensors shards
and a manifest linking every shard to the source checkpoint SHA256. The manifest
is written last; an interrupted export cannot be used as a complete artifact.
The original file is opened read-only. No pickle-based checkpoint is created.

From the repository root, in the existing Linux CUDA environment:

```bash
python -m cosmos3_h_dreams.fp8_weights \
  --checkpoint assets/c3-h-d_2-step_model_bf16.safetensors \
  --output artifacts/dreams-generator-mlp-fp8-v1 --device cuda
```

This creates an **overlay**, not a standalone replacement for `--checkpoint`.
Keep the original checkpoint available. To enable it in replay, browser or token
streaming commands, add:

```bash
--fp8-weights artifacts/dreams-generator-mlp-fp8-v1
```

Or set `runtime.fp8_weights: artifacts/dreams-generator-mlp-fp8-v1` in session YAML.
Omit the argument, or pass `--no-fp8-weights` to override YAML and return to BF16.
The loader checks source identity, shard integrity, tensor shape, dtype and finite
positive scales before replacing the MLPs, before session compilation/graph capture.
The overlay is an experimental derivative; do not publish it before acceptance.

## Matched generation and quality trial

Use an otherwise idle GPU, the same checkpoint, initial image, controls, decoder,
seed, eight output frames per chunk and two steps in both processes. Keep decoder
selection identical to deployment. These commands use the repository demo's
TAEHV decoder and previously retained cache optimizations; change `--decoder` in
**both** commands if deployment uses WAN or TAEHV2Super. Existing model/VAE assets
and the repository's media dependencies must already be installed/cached.

First check real-shape eager, compile and CUDA graph execution. This is a kernel
correctness check, not evidence of full-model quality or speed:

```bash
python -c "import runpy; runpy.run_path('cosmos3_h_dreams/tests/test_fp8_cuda.py')['test_fp8_real_shapes_compile_and_graph']()"
```

Run each variant in a fresh process with its own new compiler-cache directory:

```bash
TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/fp8-trial/cache-bf16" \
python -m cosmos3_h_dreams.cache_refresh_probe \
  --checkpoint assets/c3-h-d_2-step_model_bf16.safetensors \
  --mode compiled --reuse-cache-rope --defer-cache-refresh \
  --decoder taehv --seed 11 --frames 512 --save-video \
  --output artifacts/fp8-trial/bf16

TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/fp8-trial/cache-fp8" \
python -m cosmos3_h_dreams.cache_refresh_probe \
  --checkpoint assets/c3-h-d_2-step_model_bf16.safetensors \
  --mode compiled --reuse-cache-rope --defer-cache-refresh \
  --decoder taehv --seed 11 --frames 512 --save-video \
  --fp8-weights artifacts/dreams-generator-mlp-fp8-v1 \
  --output artifacts/fp8-trial/fp8

python -m cosmos3_h_dreams.fp8_compare \
  artifacts/fp8-trial/bf16 artifacts/fp8-trial/fp8 \
  --output artifacts/fp8-trial/comparison.json
```

Each report records source commit, checkpoint/input hashes, GPU UUID, software,
settings, compiler-cache paths, startup, 8 warmup chunks, 32 measured chunks,
per-chunk readiness and host submission times, completed-work throughput including
final cache drain, and memory. Videos are saved outside timing. A separate run
with `--profile-stages` records synchronized stage diagnostics; the comparator
rejects those timings for headline throughput. Reset, scene-return, pending-reset
and raw-token checks must pass; videos cover 512 moving frames and 64 held-control
frames. Numeric differences cover the first 8 and 64 frames and are descriptive,
not quality acceptance thresholds. Longer autoregressive clips may diverge even
when both are plausible. View `motion.mp4` and `held-controls.mp4` side by side.

Repeat in reversed order (FP8 then BF16) with new output/cache paths, then at least
one more seed and scene before recommending the setting. Repeat on **RTX PRO 6000
Blackwell**; H100 results do not predict the same speedup. Reject a candidate that
slows completed work, produces artifacts/blotches, loses control responsiveness,
or fails reset/graph checks. Keep source weights and baseline results.

Generation throughput is frames per second of completed work. Frame readiness is
time to host RGB before the deferred cache write. Neither is control-to-screen
latency. After a demonstrated generation gain, rerun the existing browser input
sequence/presentation telemetry and retune generation lead using measured P95.
No display pacing changes are bundled into this weight experiment.

## Checks completed locally

The CPU artifact tests exercise separate-file roundtrip, unchanged source bytes,
selection of only generator MLPs, zero-row scaling, wrong-source and corrupt-shard
rejection, malformed scales, and the operand/scaling contract with an explicit CPU
GEMM oracle. On a Mac without Triton, these run with only CUDA kernel imports
replaced by fail-fast placeholders; CUDA operations are not validated there.
The manual CUDA test and full model comparison remain required.
