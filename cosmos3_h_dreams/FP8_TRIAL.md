# Experimental FP8 generator MLPs for RTX PRO 6000 Blackwell

Status: **Not recommended for H100 speed; experimental memory-saving candidate.**
Full-model H100 compilation and repeated generation passed, but this FP8 path
was 2.4% slower than BF16 while saving 5.1 GiB of resident allocated memory.
RTX PRO 6000 is untested. Default: **off**. No control-to-screen reduction is claimed.
This branch adds a separate FP8 weight overlay and runtime path; it does not change
any published checkpoint or the shared `flashdreams/` implementation.

## Decision

Start with FP8 E4M3 weights and dynamic per-token FP8 activations in the generator
MLPs. This targets dense matrix multiplication in both denoising and clean-cache
refresh. All 36 generator MLPs (108 projections) are eligible. Gate and up share
one activation quantization; down quantizes its own input. Scaling is per output
channel for weights and per row for activations. GEMM outputs stay BF16. SwiGLU
uses an explicit FP32 SiLU/product and rounds to BF16 before down-projection
quantization; the existing GEMM uses its default full accumulation.
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
  --decoder taehv --seed 11 --frames 512 --timed-chunks 128 --save-video \
  --output artifacts/fp8-trial/bf16

TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/fp8-trial/cache-fp8" \
python -m cosmos3_h_dreams.cache_refresh_probe \
  --checkpoint assets/c3-h-d_2-step_model_bf16.safetensors \
  --mode compiled --reuse-cache-rope --defer-cache-refresh \
  --decoder taehv --seed 11 --frames 512 --timed-chunks 128 --save-video \
  --fp8-weights artifacts/dreams-generator-mlp-fp8-v1 \
  --output artifacts/fp8-trial/fp8

python -m cosmos3_h_dreams.fp8_compare \
  artifacts/fp8-trial/bf16 artifacts/fp8-trial/fp8 \
  --output artifacts/fp8-trial/comparison.json
```

Each report records source commit, checkpoint/input hashes, GPU UUID, software,
settings, compiler-cache paths, startup, 8 warmup chunks, 128 measured chunks
(the CLI default is 32),
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
The manual CUDA test passed on H100 after the precision correction below.
The full-model H100 comparison is recorded below; RTX PRO 6000 checks remain required.

The first H100 pilot (job 6481804) passed all 55 native CPU tests and executed
FP8 GEMM, then stopped at the eager/compile parity check before weight export.
For the one-row probe, 45/4096 elements exceeded the existing 0.03 absolute/relative
tolerance (maximum absolute difference 0.0510). Generated Inductor source showed
the intermediate BF16 SiLU cast was removed by fusion. The follow-up makes the
activation's FP32 computation and final BF16 rounding explicit in both execution
modes; the original test tolerance is unchanged. This diagnosis is not full-model
or target-GPU validation.

The corrected H100 kernel check (job 6481830) was bit-exact between eager and
compile at all four probe sizes, with exact CUDA graph replay. It exported all
36 FP8 shards. Full-model BF16 generation ran, but video export stopped because
the container lacks optional `mediapy`; the probe now uses its existing FFmpeg.
These failed allocations remain included in the compute ledger. The follow-up
reuses the FP8 export and a copied BF16 compiler cache; startup times from that
bounded comparison are not a matched cold-start benchmark.

Job 6481852 verified and installed all 108 FP8 projections in the full model,
then hit its 125-second worker timeout during compile warmup. That attempt produced
no FP8 steady-state timing or quality videos. Its partial compiler cache was
preserved and reused in the completed comparison below. Original weights and
default BF16 behavior remain the recommended path for speed.

Pilot accounting: jobs 6481804/6481830/6481852 used 120/269/184 node-seconds,
including the failed attempts. Before the completed comparison, cumulative usage
was 2,637 node-seconds. The user then lifted the prior compute cap to finish
compilation and repeated generation. Each allocation reserved eight H100s.
The real FP8 overlay is 5,439,968,064 bytes and represents 10,871,635,968 bytes
of BF16 MLP matrices.

## Completed repeated H100 comparison — 2026-11-07

Job **6481895**, source **dc565d362cdf04eb9f64acf826cd34efde0a93a8**, completed
all four fresh processes in FP8/BF16/BF16/FP8 order on the same H100 80 GB.
Each used eight warmup chunks followed by 128 measured chunks: 2,048 measured
frames per precision. Both retained the latest cache optimizations, 832×480 output,
eight frames/chunk, two denoising steps, TAEHV, and seed 11. PyTorch was
2.11.0+cu130 with CUDA 13.0. This compares FP8 against the optimized BF16 runtime,
not against the original release.

| Metric | BF16 | Selective FP8 |
| --- | ---: | ---: |
| Complete generation per eight-frame chunk, mean | 117.21 ms | 120.06 ms |
| Completed generation throughput | 68.25 fps | 66.64 fps |
| Host RGB readiness, median / P95 | 69.17 / 70.00 ms | 71.60 / 72.36 ms |
| GPU memory allocated after timing | 29.85 GiB | 24.75 GiB |
| GPU memory reserved after timing | 31.45 GiB | 26.15 GiB |
| Peak allocated memory over the process | 36.58 GiB | 33.88 GiB |

The two BF16 process means were 116.94 and 117.49 ms/chunk; FP8 was 119.76
and 120.35 ms/chunk. FP8 increased generation time by **2.42%** and reduced
throughput by **2.37%**. Median frame readiness was **2.43 ms later**. Completed
generation includes decode, host RGB preparation, cache refresh and final GPU
drain. No browser, transport or display latency was measured. Peak memory includes
startup and subsequent reset/quality checks; it is not the steady-state allocation.
These are PyTorch allocator counters; CUDA context and other non-PyTorch memory
are not included.

All finite-output, reset, scene-return, pending-reset and raw-token checks passed.
The two runs of each precision produced exactly identical 512-frame hash sequences.
BF16 and FP8 are not pixel-identical: first-eight-frame moving-clip PSNR was
34.09 dB; first-64-frame PSNR was 28.49 dB. These are descriptive differences,
not evidence of quality equivalence. Saved quality clips contain 512 moving and
64 held-control frames per run (51.2 and 6.4 seconds at the preset's 10 fps).
This is one scene and one seed; broader quality and actual RTX testing are needed.
Visual spot checks at moving frames 0/63/127/255/383/511 and held-control frames
0/7/15/31/47/63 found recognizable scene structure in both variants, with tissue
position and texture differences but no obvious additional FP8 blotchy collapse
in those samples. This is not a full temporal-quality or control-response verdict.
Both full moving and held-control pairs are saved for human side-by-side review.

Persistent compiler caches were copied from prior attempts. Initialization took
32.58/31.53 seconds for BF16 and 93.23/62.20 seconds for FP8, including load and
initial compile work. These are not matched cold-start measurements. The job
used 541 node-seconds, or 1.202 allocated H100-hours across the eight-GPU node.
Measurements ran sequentially on GPU 0.

## Why native FP8 did not improve H100 speed

A separate diagnostic allocation, job **6481930**, completed both precisions on
another H100 with the same source, settings and copied compiler caches. After
warmup it recorded eight synchronized stage samples and four profiler chunks per
precision. These instrumented samples explain the result; the repeated trial
above remains the generation benchmark.

The trace confirms actual FP8 matrix kernels, with no BF16 precision fallback.
Summed dense GEMM kernel time was 44.69 ms/chunk for BF16 and 43.84 ms/chunk for
FP8, a saving of only **0.85 ms**. The separate dynamic activation quantizer kernels
added **3.12 ms/chunk**. The GEMM group includes unchanged projections, and GPU
kernel-duration sums are not end-to-end wall time. Synchronized denoising medians
were 48.53/50.49 ms for BF16/FP8; cache-refresh submission was 49.24/48.47 ms.

This implementation's activation-conversion cost outweighs its measured matrix
saving. Additional compile warmup will not remove that steady-state work. A next
FP8 experiment should benchmark fused producer/quantizer kernels or a more
efficient scaled GEMM implementation at the model's real shapes before repeating
full rollouts. No runtime change is bundled with this diagnosis, and the same
outcome or speedup should not be assumed on RTX PRO 6000 Blackwell.

The diagnostic used 213 node-seconds (0.473 allocated H100-hours). Together with
the repeated trial, this continuation consumed **754 node-seconds / 1.676 allocated
H100-hours**. Including all earlier weight-quality and FP8 attempts, the ledger
stands at 3,391 node-seconds / 7.536 allocated GPU-hours. Both jobs completed,
and no GPU job remains active for this FP8 continuation.
