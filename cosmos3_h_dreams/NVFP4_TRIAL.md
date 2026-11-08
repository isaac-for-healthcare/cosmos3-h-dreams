# Experimental NVFP4 generator MLPs on RTX PRO 6000D

Status: **Useful opt-in for evaluation; experimental, default off.** See the
[reproduction guide](experiments/nvfp4/README.md) for export and benchmark commands.
The packaged
application is faster than FP8 on the tested RTX PRO 6000D. Visual quality
acceptance remains open; NVFP4 has larger numerical differences from BF16 than
FP8. Do not promote it as the default. The standard RTX PRO 6000 has not been
measured.

## What changes

Only the 36 generator MLPs (108 projections) use NVFP4. Both weights and
activations have E2M1 payloads, FP8 E4M3 scales per 16 values, and an FP32 global
scale per tensor. Accumulation is FP32 and output is BF16. Gate and up share
activation quantization. SwiGLU rounds once to BF16 after its FP32 SiLU/product.
Attention, action conditioning, understanding weights, readout and VAE keep their
existing precision. Eight-frame chunks and two denoising steps are unchanged.

The implementation uses the installed PyTorch `aten._scaled_mm_v2` operator and
a vectorized Triton converter. No custom GEMM library or new dependency is
needed. Exact packing/scaling checks include padded rows, zeros and dynamic
ranges. PyTorch's GEMM matched the separate CUTLASS reference exactly on the
tested shapes. The vectorized converter reduced common-shape conversion time
from about 18/49 microseconds to 8/18 microseconds for gate/down inputs.

Validated prototype stack: one RTX PRO 6000D Blackwell Workstation Edition,
SM120, approximately 72 GiB; driver 615.71.09; PyTorch 2.11.0+cu130; Triton 3.6.0;
cuDNN 91900. The runtime currently requires SM120 and performs an actual kernel
preflight. Missing support fails clearly without a precision fallback. Private
PyTorch operator compatibility must be rechecked after upgrades.

## Prototype generation results — 2026-11-08 UTC

Two fresh processes per precision, in forward and reverse order. Each process
used eight warmup chunks and 128 measured chunks, then a 512-frame moving
rollout and a 64-frame held-control rollout outside timing. Source runtime:
`7e6801c32d1be84c8b5ce3b485df1b6c7e2052cf`, with a separately hashed NVFP4 prototype.
Seed 11, repository demo image/actions, 832×480, TAEHV decoder, compiled denoise
and cache refresh, deferred refresh and positional-factor reuse.

| Precision | Complete generation, ms / 8 frames | Generated FPS | Median RGB-ready, ms | Allocated VRAM, GiB |
|---|---:|---:|---:|---:|
| Optimized BF16 | 230.10 | 34.77 | 143.58 | 29.78 |
| Optimized FP8 | 188.93 | 42.34 | 116.18 | 24.68 |
| Experimental NVFP4 | 155.85 | 51.33 | 94.61 | 22.53 |

NVFP4 used **17.51% less generation time** than FP8, equivalent to **21.23% more
throughput**, and 2.15 GiB less allocated memory. Relative to optimized BF16,
generation time fell 32.27%. These are paired full-generation results, not
projection-kernel speedups or physical display latency. Complete time includes
deferred cache work and the final GPU drain; RGB readiness excludes the pending
refresh. Allocated memory is neither reserved memory nor startup peak memory.

Both repeats generated identical videos within each precision. All finite-output,
reset, scene-return, pending-reset, raw-token and close/drain checks passed.
The original source checkpoint SHA256 is
`3e6fce0046a6fe46c65307245df0db041c4a1828e0b3fc76708fb30e7851c97b`.

## Packaged generation confirmation

The actual product flag and exporter were then checked without the prototype
loader patch. Four fresh processes ran NVFP4 / FP8 / FP8 / NVFP4, each with eight
warmup chunks, 128 timed chunks, a 512-frame moving rollout and 64 held-control
frames. All other generation settings matched the prototype comparison. Each
precision reused its validated compiler cache. The source overlay is recorded in
`nvfp4-package-01/inputs/source-sha256.json`; these are working-tree additions to
the pinned runtime, not an upstream release claim.

| Packaged precision | Complete generation, ms / 8 frames | Generated FPS | Allocated VRAM, GiB |
|---|---:|---:|---:|
| FP8 | 188.88 | 42.35 | 24.68 |
| NVFP4 | 154.68 | 51.72 | 22.53 |

This confirms **18.11% less generation time**, **22.11% more throughput**, and
**2.15 GiB less allocated memory** versus FP8. Per-process complete times were
188.49/189.27 ms for FP8 and 154.24/155.12 ms for NVFP4. Both repeats matched
their same-precision reference videos exactly. Finite output, reset, scene
return, pending reset, raw-token output and close/drain checks passed.
NVFP4's measured peak allocation was 31.22 GiB; the lower resident figure does
not describe startup requirements.

## Packaged software control-to-browser results

Headless Chromium on Colossus loopback, MJPEG, 30 FPS presentation, the same
irregular keyboard press/release schedule and eight-frame model chunks. Two
75-second trials per configuration, with seconds 15–70 measured: 142 control
changes per configuration. Reverse-order repeats limit ordering effects.

| Configuration | Generation lead, ms | Median control-to-browser, ms | P95, ms |
|---|---:|---:|---:|
| FP8 | 140 | 286.00 | 423.25 |
| NVFP4, adjusted scheduling | 115 | 261.60 | 398.46 |

With adjusted scheduling, the median improved by **24.40 ms (8.53%)** and P95 by
**24.80 ms (5.86%)** versus FP8. Every measured control was received, consumed
and observed. No missing frame IDs or presentation gaps above 100 ms occurred.

An earlier prototype sweep separated precision from scheduling: at the same
140 ms generation lead, median/P95 changed from 281.35/418.04 ms with FP8 to
270.65/415.56 ms with NVFP4. Reducing NVFP4's lead to 115 ms produced
255.75/373.34 ms. These earlier numbers are a separate trial, not the packaged
headline. The repeats show a consistent median benefit; the tail benefit varies
across runs. This is a small, one-scene benchmark, not a general latency guarantee.

The endpoint is browser WebSocket control send to the first corresponding decoded
image observed at `requestAnimationFrame`. Chunk lineage establishes which
controls conditioned the frame; it does not establish visible tool-motion onset.
This excludes remote network, headset hardware and physical display scanout.
The generation and latency percentages must not be added together.

## Quality limits

The first 8 moving frames had PSNR 28.07 dB against optimized BF16 for NVFP4,
versus 29.33 dB for FP8. Over the first 64 frames these were 24.16 and 27.38 dB.
On identical real BF16 rollout inputs, median MLP relative RMSE was 15.5% for
NVFP4 versus 4.45% for FP8. These are diagnostics, not quality acceptance scores.
Long autoregressive rollouts diverge and cannot be interpreted as pixel-aligned
ground truth. Review instrument shape, control response, drift and blotchiness
in the saved videos before choosing this tradeoff.

Two additional scenes (2115 and 2366) were tested with held and pulsed controls,
512 frames each, seed 17 and matched captions. All 12 prototype cases across
BF16/FP8/NVFP4 passed finite-output, reset and shared-control-prefix checks. All
four packaged NVFP4 cases also passed. The schedule uses 256 frames of synthetic
controls followed by 256 frames holding the final control; this is not recorded
live surgical telemetry. Contact-sheet inspection found no obvious blotchy
collapse, but showed visible instrument/tissue geometry and drift differences,
especially in scene 2366. Visual quality equivalence is not established. The
side-by-side videos retain all frames and use a common 30 FPS presentation rate.

## Validation and compilation reproducibility

The Linux CPU suite passed 188 tests. Package build and lock-file validation
passed. The separate GPU check exercised every FP4 code, zero inputs, an
independently decoded GEMM reference, real MLP shapes, compilation and exact CUDA
graph replay. The packaged exporter reproduced all 36 prototype shard hashes.

Packaging first failed a whole-video equality check against the original
prototype cache. Isolated eager and compiled MLPs were bit-exact between
prototype and product on six saved real inputs (layers 0/17/35, chunks 0/1).
Recompiling the unchanged prototype into a fresh compiler cache also changed
its output. Controlled 32-versus-128 timed-chunk checks remained frame-exact
within each cache, ruling out the changed timed-loop length in these tests.

Thus the final gate uses exact same-cache repeats and reference comparisons;
it does not claim bitwise reproducibility across independent compiler/autotune
caches. The specific whole-graph kernel choice responsible has not been isolated.
Retain the recorded compiler caches when reproducing a particular video, and
repeat quality checks after recompilation or a software upgrade.

For scale, the first eight frames of the unchanged prototype after recompilation
had PSNR 29.00 dB against its original cache; packaged versus original prototype
was 27.88 dB. These differences are material enough to preserve the compiler
cache with each reference video, rather than treating all differences as packing
error or silently accepting a failed reference check.

CUDA trace attribution now places attention at roughly 55.6 ms per chunk and
NVFP4 packing at 2.45 ms per chunk, excluding the separate reduction kernels.
Further packing work has limited total-generation upside. These are kernel-trace
attributions; do not sum profiler parent and child operator entries.

## Separate export, usage and rollback

The original BF16 checkpoint remains required and is opened read-only. The
exporter writes a new directory with 36 safetensors shards and a manifest linking
every shard to the original checkpoint hash. It refuses to overwrite an existing
directory and publishes the manifest only on successful completion.

```bash
python -m cosmos3_h_dreams.nvfp4_weights \
  --checkpoint assets/c3-h-d_2-step_model_bf16.safetensors \
  --output artifacts/dreams-generator-mlp-nvfp4-v1 --device cuda
```

Add `--nvfp4-weights artifacts/dreams-generator-mlp-nvfp4-v1` to a supported demo
command, or set `runtime.nvfp4_weights` in session YAML. FP8 and NVFP4 flags are
mutually exclusive. Omit both overlays for BF16; `--no-nvfp4-weights` overrides
an NVFP4 choice in YAML. The loader validates every shard before replacing any
MLP. Initial loading still stages the original BF16 model, so resident savings
do not describe startup peak memory.

Run the target-GPU check, then a full matched trial:

```bash
python -c "import runpy; runpy.run_path('cosmos3_h_dreams/tests/test_nvfp4_cuda.py')['test_nvfp4_real_shapes_compile_and_graph']()"

python -m cosmos3_h_dreams.cache_refresh_probe \
  --checkpoint assets/c3-h-d_2-step_model_bf16.safetensors \
  --mode compiled --reuse-cache-rope --defer-cache-refresh \
  --decoder taehv --seed 11 --frames 512 --timed-chunks 128 --save-video \
  --nvfp4-weights artifacts/dreams-generator-mlp-nvfp4-v1 \
  --output artifacts/nvfp4-trial/nvfp4
```

Compare with the same command using `--fp8-weights` and a separate output/cache
directory, and again with neither overlay for BF16. Keep all other flags and
inputs identical. Use `--reference-quality` only for same-precision deterministic
parity, not to demand equality between different quantization formats.

## Attention follow-up: benchmark-only experiment, 2026-11-08

The next experiment kept the packaged NVFP4 weights, eight output frames, two
denoising steps, 832×480 resolution, TAEHV decoder, and history semantics fixed.
It ran on the same RTX PRO **6000D** workstation, torch 2.11.0+cu130, cuDNN 91900,
Triton 3.6, driver 615.71.09. These are additional gains over packaged NVFP4,
not a fresh comparison against production BF16 or a standard RTX PRO 6000.

| Attention path | Complete generation, ms / 8 frames | Generated FPS | Generation time reduction | RGB readiness p50 across repeats |
|---|---:|---:|---:|---:|
| Existing NVFP4 attention | 154.90 | 51.65 | reference | 94.30–94.35 ms |
| cuDNN SDPA | 142.71 | 56.06 | 7.87% | 93.75–94.05 ms |
| Temporal split FlashAttention + native GQA | 135.90 | 58.87 | 12.27% | 94.36–94.40 ms |

Complete generation includes deferred cache refresh and the final GPU drain.
Each row averages two fresh-process runs, each with eight warmup chunks and 128
timed chunks using its warmed compiler cache. Default/cuDNN used ABBA ordering;
the two split/GQA repeats followed the candidate screen. Split/GQA saves 19.00 ms
per chunk and increases generated throughput by 13.98%. The percentages describe
the same generation improvement with different denominators; do not add them.
Steady allocated memory remained 22.53 GiB. **No new control-to-screen latency
improvement was established:** RGB readiness stayed near 94 ms, and browser
latency was not rerun for this attention change.

The captured saturated refresh mask has two 394-token temporal blocks. Splitting
it into two mask-free calls preserves visibility: each block sees all cached
history and the permitted current tokens. Generated compiler code confirms
FlashAttention calls with query length 394 and key lengths 5165 and 5559. Text
prefill and padded cache fill retain their original paths. The microbenchmark
used real layer-0 inputs and an independent FP32 math-attention reference; all
supported candidates passed relative RMSE <0.005. Mask-equivalence CPU checks
also passed. The kernel screen normalized layout to contiguous tensors; the
full-model runs retained actual compiled layouts.

Split attention without native GQA screened at 136.37 ms/chunk, versus 135.97 ms
with GQA. That small difference does not establish an independent GQA benefit.
Globally enabling native GQA with the original masked refresh fell back to math
attention and was substantially slower in the kernel screen. The earlier split
attempts failed on text prefill and padded cache fill; those failures and their
frozen inputs are retained with the evidence.

The cuDNN repeats matched all 512 reference frames within their cache; split/GQA
matched all 320 reference frames in both repeats. Finite-output, reset,
scene-return, pending-reset, and raw-output lifecycle checks passed. Each also
saved 64 held-control frames. These checks establish repeatability, not equality
with the original NVFP4 video. Against that baseline, first-eight-frame PSNR was
30.00 dB for cuDNN and 25.61 dB for split/GQA; over the first 64 moving frames it
was 25.39 and 23.75 dB. These are difference diagnostics, not quality scores.
Attention arithmetic and independently compiled caches both can change the
autoregressive trajectory. Sampled contact sheets show instrument/tissue geometry
differences without obvious blotchy collapse. Quality equivalence is unproven.

The prototype remains outside product source and default-off. Its video routing
is restricted to the tested geometry/mask; it is not a generic production
implementation. Before packaging, derive the routing from runtime mask metadata,
retain a supported fallback for other masks/shapes, and repeat multi-scene quality
and browser latency checks. The original BF16 and NVFP4 artifacts are unchanged;
this experiment changes attention execution, not weights.

Frozen scripts, failures, real Q/K/V tensors, candidate compiler caches, raw
timings, and videos are preserved in the SHA256-verified Eos archive at
`/lustre/fsw/healthcareeng_holoscan/user_data/shuver/outputs/dreams-attention-20261108/evidence-01`.
Baseline source, weights, and baseline compiler caches remain in the preceding
`dreams-nvfp4-20261108` evidence set. Archive symlinks retain original Colossus
paths and must be remapped when restoring elsewhere. This benchmark did not
change production defaults; the scoped follow-up tools are now included below.


## Scoped refresh validation and presentation timing — 2026-11-08

A clean-lease reproduction on the same physical RTX PRO **6000D** recovered the
prior results using the saved compiler caches. The revised benchmark changes
only saturated compiled FadeMem refresh; it preserves text prefill, image/seed
processing, cache filling, denoising and the existing repeated KV heads. Runtime
mask/token metadata replaces the earlier hard-coded geometry. It uses the
existing model-factory hook; product and `flashdreams/` source remain unchanged.
This is a refinement of the preceding attention gain, not another additive gain.

| Configuration | Complete ms / 8 frames | Generated FPS |
|---|---:|---:|
| Existing packaged NVFP4 | 155.09 | 51.58 |
| Refresh-only split attention | 136.50 | 58.61 |

Two fresh processes per setting, eight warmup and 128 timed chunks each, yielded
**18.59 ms saved: 11.98% less generation time / 13.62% more throughput**. RGB
readiness remained near 94 ms, with 22.53 GiB steady allocated memory. Original
BF16 and all 36 separate NVFP4 shards are unchanged. Same-candidate repeats,
finite output, reset, scene return, pending reset and raw-output checks passed.
A 640×384 smoke test with a different caption and seed also passed; its timings
are excluded from this 832×480 table. Production integration should use an
explicit helper/parameter rather than the scoped function-copy benchmark.

Eight 1,024-frame clips compare two scenes and held/pulsed controls. The first 56
frames match the existing NVFP4 baseline exactly in all four pairs; later output
diverges. Six early/late contact sheets show visible instrument/tissue geometry
differences without obvious blotchy collapse in sampled frames. These remain
quality smoke tests, not equivalence acceptance. Full side-by-side videos retain
all 1,024 frames at 30 FPS. Review them before integrating or promoting the option.

Two 75-second trials per setting, 142 measured control changes, established:

| Experiment | Median before → after, ms | P95 before → after, ms |
|---|---:|---:|
| Existing → refresh-only attention, same 30 FPS / 115 ms lead | 261.65 → 256.55 | 401.74 → 384.67 |
| 115 → 105 ms lead, refresh-only attention / 30 FPS | 256.55 → 266.75 | 384.67 → 395.81 |
| 30 → 45 FPS target, refresh-only attention / 115 ms lead | 263.65 → 233.20 | 389.05 → 311.55 |
| Existing → refresh-only attention, same 45 FPS / 115 ms lead | 231.65 → 232.20 | 310.85 → 314.26 |

At 30 FPS the attention-only median difference is small (5.10 ms); the shorter
105 ms lead regressed and is rejected. Increasing presentation to 45 FPS saved
**30.45 ms median (11.55%) and 77.51 ms P95** in its paired trial. A final matched
45 FPS comparison found **no demonstrated attention-only latency improvement**:
existing NVFP4 already sustained approximately 43.4 browser-observed FPS. Thus
higher presentation rate provides the larger latency win; attention provides
additional generation headroom. Do not add the percentages or attribute the
presentation benefit to the new attention path.

All measured controls were received, consumed and observed; no frame IDs were
missed and no presentation gaps exceeded 100 ms. Endpoint, loopback MJPEG,
headless Chromium and physical-display limitations are as described above.
Keyboard velocity was normalized per wall-clock second between presentation
rates; model FPS stayed 10. Presentation rate still changes action sampling,
simulation presentation speed and rollout depth per wall time. A 45 FPS target
requires a physical-display cadence check; it does not establish that jerkiness
is solved on a 60 Hz display. Standard RTX PRO 6000 remains unmeasured.

Status: **useful opt-in for evaluation; benchmark-only and default-off**.
Integration remains deferred pending video review and target-hardware validation.
The next latency test can use a display-appropriate higher presentation target
with the existing NVFP4 path. The published implementation remains default-off,
with benchmark-only refresh/presentation tools in `experiments/nvfp4`.
Frozen inputs, failed/cancelled attempts, raw timings, all videos and actual
compiler caches are retained in the verified Eos evidence set:
`/lustre/fsw/healthcareeng_holoscan/user_data/shuver/outputs/dreams-attention-validation-20261108/evidence-01`.
