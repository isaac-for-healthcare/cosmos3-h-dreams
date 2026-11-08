# Experimental NVFP4 and refresh-attention benchmarks

These source-checkout tools reproduce the generation and local browser
experiments described in [NVFP4_TRIAL.md](../../NVFP4_TRIAL.md). They are not
production defaults. Eight output frames per chunk and two denoising steps
remain fixed. The NVFP4 product option and exporter are separate from the
benchmark-only refresh-attention choice.

Validated hardware: RTX PRO **6000D** Blackwell Workstation Edition (SM120),
driver 615.71.09, Python 3.12.16, torch 2.11.0+cu130, cuDNN 91900, Triton 3.6.0.
The standard RTX PRO 6000 is a validation target, not a measured result.
Use the repository's locked Linux environment and run commands from its root.

## Export separate weights

Obtain the original [Cosmos3-H-Dreams checkpoint](https://huggingface.co/nvidia/Cosmos3-H-Dreams)
at revision `fd87afe3fd97e9b352b09d9ada88c944f5621eb9`, path
`checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors`.
Its SHA256 is `3e6fce0046a6fe46c65307245df0db041c4a1828e0b3fc76708fb30e7851c97b`.
Use your local checkpoint path below. The exporter reads it without changing it
and refuses to overwrite an existing output directory. No weights ship in Git.

```bash
export PYTHONPATH="$PWD/cosmos3_h_dreams:$PWD/flashdreams"
export DREAMS_CHECKPOINT="$PWD/assets/c3-h-d_2-step_model_bf16.safetensors"
export DREAMS_NVFP4_WEIGHTS="$PWD/artifacts/nvfp4-v1"

.venv/bin/python -m cosmos3_h_dreams.nvfp4_weights \
  --checkpoint "$DREAMS_CHECKPOINT" --output "$DREAMS_NVFP4_WEIGHTS"
```

The original checkpoint is still required when loading the 36-shard overlay.
Add `--nvfp4-weights "$DREAMS_NVFP4_WEIGHTS"` to a supported application command.
Omit both precision-overlay flags for BF16. FP8 and NVFP4 are mutually exclusive.
NVFP4 currently requires the tested SM120 kernel path and fails explicitly when
it is unavailable; it does not silently use another precision.

## Generation and reset/rollout checks

Use your host's shared GPU lock to exclude concurrent GPU workloads. Keep the
baseline and candidate compiler caches separate and retain them with results.
Run the following once for `default` and once for `refresh-split`, then repeat
in reverse order with new output directories. The generation harness performs
eight warmup chunks before measuring 128 complete chunks.

```bash
# Change these for each fresh process. Never overwrite an earlier run.
ATTENTION=default
RUN_NAME=default-01
export TORCHINDUCTOR_CACHE_DIR="$PWD/artifacts/nvfp4-caches/$ATTENTION/torch"
export TRITON_CACHE_DIR="$PWD/artifacts/nvfp4-caches/$ATTENTION/triton"
mkdir -p artifacts

flock /path/to/shared/gpu.lock timeout --kill-after=15s 1800 \
  .venv/bin/python cosmos3_h_dreams/experiments/nvfp4/generation.py \
  --attention-variant "$ATTENTION" \
  --checkpoint "$DREAMS_CHECKPOINT" --nvfp4-weights "$DREAMS_NVFP4_WEIGHTS" \
  --mode compiled --reuse-cache-rope --defer-cache-refresh \
  --seed 11 --frames 1024 --timed-chunks 128 --save-video \
  --output "artifacts/nvfp4-generation/$RUN_NAME"
```

For a same-candidate repeat, add `--reference-quality` pointing to that
candidate's first `quality.json`. Do not demand frame equality between different
precision or attention choices. Fresh compiler/autotune caches can change an
autoregressive rollout; rerun quality checks after rebuilding them. First 56
frames matching the baseline does not establish later-rollout equivalence.

The refresh prototype derives its temporal blocks from runtime mask metadata.
It rebinds attention only in the saturated compiled refresh callable, retaining
the original layer code and all other paths. It uses `model_factory`; it does
not edit `flashdreams/` or globally replace product attention. For production
integration, replace the scoped function-copy technique with an explicit helper.

## Local browser timing

The optional `benchmark-browser` development group pins Playwright. The commands
below install that same version separately from the locked model environment.
Linux needs the
browser's system dependencies, GNU `timeout` and the `flock` utility above.

```bash
uv pip install --python .venv/bin/python \
  --target artifacts/browser-tools/python playwright==1.64.1
export PYTHONPATH="$PWD/artifacts/browser-tools/python:$PYTHONPATH"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/artifacts/browser-tools/browsers"
.venv/bin/python -m playwright install chromium
```

Warm both attention caches with the generation commands first. Commit source
changes before measuring: the runner records the actual commit, copies its
protocol and hashes the benchmark inputs. It checks the checkpoint identity,
acquires the supplied GPU lock and starts fresh bounded processes in the
specified order. Incomplete runs retain logs and an unsuccessful `outcome.json`.

```bash
.venv/bin/python cosmos3_h_dreams/experiments/nvfp4/matrix.py rate \
  --checkpoint "$DREAMS_CHECKPOINT" --nvfp4-weights "$DREAMS_NVFP4_WEIGHTS" \
  --cache-root artifacts/nvfp4-caches --gpu-lock /path/to/shared/gpu.lock \
  --output artifacts/nvfp4-browser-rate-01
```

Append `--dry-run` to inspect the matrix without loading a model or writing a run.
The same command supports three protocols:

| Protocol | Comparison | Trials |
|---|---|---:|
| `latency30` | Existing vs refresh attention at 115 ms lead; candidate at 105 ms lead | 6 |
| `rate` | 30 vs 45 FPS target, refresh attention, 115 ms lead | 4 |
| `attention45` | Existing vs refresh attention at 45 FPS, 115 ms lead | 4 |

Each trial runs for 75 seconds after startup; seconds 15–70 measure 71 irregular
control changes. Each setting repeats twice. Inspect `summary.json`, per-trial
client/server traces and missed-frame counts. Analysis requires every measured
control to be received, consumed and observed. Commands reject existing output
directories. `--worker-timeout` defaults to 420 seconds including initialization.

The endpoint is synthetic keyboard-state WebSocket send to the first corresponding
decoded MJPEG frame observed at browser rAF, all on server loopback. It does not
measure physical input/display latency or visible tool-motion onset. Model FPS
remains 10; keyboard velocity is normalized between presentation rates. Other
motion, rollout depth and display cadence still change with presentation rate.
The keyboard adapter is instrumentation for this benchmark, not a new application
entry point. The 45 FPS target is experimental, not a default recommendation.

## Validation and observed results

```bash
# Standard-library protocol and trace-accounting checks (no GPU/model imports).
python3 cosmos3_h_dreams/experiments/nvfp4/selfcheck.py

# Uses the model environment; checks mask equivalence and fallback on CPU.
PYTHONPATH=cosmos3_h_dreams:flashdreams \
  .venv/bin/python cosmos3_h_dreams/experiments/nvfp4/attention_variants.py
```

Historical Colossus checks covered the NVFP4 product tests (188 Linux CPU tests),
packing/GEMM correctness, compilation, CUDA graph replay, reset/scene switching,
repeat generation and longer rollouts. The published runtime files match those
tested files. The benchmark packaging adapts paths, matrix launching and metadata;
the attention computation, control schedule and frame-lineage calculation are
retained. The portable launcher itself has not been rerun on a GPU after packaging.
Packaging checks passed the matrix dry runs, protocol/stage-accounting checks,
both browser JavaScript suites, dependency-policy checks, lock validation and
wheel/source builds. Replaying the packaged analyzer on the archived rate and
matched-45-FPS traces reproduced every combined result exactly. The full Python
suite could not collect on the packaging Mac because Triton is unavailable;
the earlier 188-test result is from Linux, not a fresh Mac run.

On the measured 6000D, existing NVFP4 vs refresh-only attention took 155.09 vs
136.50 ms per eight frames (11.98% less generation time). At 30 FPS, the median
browser difference was small: 261.65 vs 256.55 ms. Changing the presentation
target from 30 to 45 FPS saved 30.45 ms median in a separate paired trial.
Existing NVFP4 already reached about the same latency at 45 FPS as the refresh
candidate (231.65 vs 232.20 ms). Do not add these percentages or attribute the
presentation benefit to attention. The 105 ms lead regressed and was rejected.
Quality equivalence remains unproven; see the full trial report.
