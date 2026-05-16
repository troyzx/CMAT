# Benchmarking

CMAT now has two practical benchmark surfaces:

1. the astronomy-specific reduced regression benchmark in [`docs/rebuild/WASP44_REDUCED_BENCHMARK.md`](rebuild/WASP44_REDUCED_BENCHMARK.md), which protects scientific behavior without requiring a production-scale rerun;
2. the Stage 5 simulator-adapter examples, which are small enough to use for runtime, multiprocessing, and artifact-layout comparisons.

This guide focuses on the second surface: how to record runtime, grid size, multiprocessing strategy, and memory use in a way that stays comparable across reruns.

## What to record

For any benchmark note, capture the same small set of facts:

- workflow surface: astronomy TTV path, generic simulator adapter, or deployment-style batch example;
- grid definition: candidate count plus the parameter ranges that produced it;
- execution controls: `worker_count`, `start_method`, and whether progress output was enabled;
- artifact path: where `run_metadata.json`, portfolio outputs, or caches were written;
- environment: Python version, dependency versions, git commit, and any cache-related environment variables.

The deployment-style adapter example already persists most of that metadata to `run_metadata.json`.

## Grid size

Benchmark notes should always state how many latent-state points were evaluated.

- astronomy TTV grid: `len(period_ratios) * len(companion_masses)`
- simulator-adapter grid: `len(parameter_grid())`

Without that number, raw wall-clock timings are hard to compare.

## Runtime expectations

Wall time is usually dominated by forward simulation, not by artifact writing.

- `run_simulator_adapter(...)` scales with the number of candidate parameter points and the cost of each `simulate(...)` call;
- portfolio and manifest writing are usually small fixed costs;
- the astronomy path adds separate TTV and MEGNO passes, so benchmark notes should state which pass was timed.

For very small grids, `worker_count=1` is often fastest because process startup can dominate the run.

## Multiprocessing strategy

`ExecutionConfig` exposes the main runtime knobs that should appear in every benchmark note:

- `worker_count`: compare at least a serial baseline and one parallel run on the same grid;
- `start_method`: use `"fork"` for simple local terminal runs on Unix-like systems, and `"spawn"` when you want cleaner worker initialization or notebook/CI-friendly behavior;
- `show_progress`: disable it for batch logs so timing output stays clean.

Do not compare worker counts across different grids or different start methods and call that a scaling result; keep one variable fixed at a time.

## Memory notes

Peak memory depends on both the workflow surface and how much intermediate state is retained.

- simulator-adapter runs keep the normalized parameter grid, score array, acceptance mask, and summary payload in memory;
- astronomy runs can also retain larger TTV, MEGNO, and posterior-sample products;
- increasing `worker_count` can increase resident memory because each worker may hold simulator state or imported dependencies.

For the astronomy path, persisted caches under `OutputConfig.cache_dir` trade disk usage for less recomputation. For the adapter path, portfolio and manifest artifacts are typically cheap relative to the simulation itself.

## Recommended command pattern

Use the constrained rebuild environment and writable cache directories:

```bash
export MPLCONFIGDIR=/private/tmp/cmat-mplconfig
export XDG_CACHE_HOME=/private/tmp/cmat-cache
```

On macOS, use `/usr/bin/time -l` to capture wall time and peak resident set size while running the deployment-style adapter example:

```bash
/usr/bin/time -l /private/tmp/cmat-rebuild-env/bin/python \
  examples/damped_oscillator_batch.py \
  --run-name bench-w1 \
  --worker-count 1

/usr/bin/time -l /private/tmp/cmat-rebuild-env/bin/python \
  examples/damped_oscillator_batch.py \
  --run-name bench-w4 \
  --worker-count 4 \
  --start-method spawn
```

Those commands keep the model and artifact layout fixed while varying only the execution controls.

## Benchmark note template

Use a short record like this:

```text
surface: damped_oscillator_batch
grid_size: 9 candidates
worker_count: 4
start_method: spawn
show_progress: false
artifacts: artifacts/deployment/bench-w4/
wall_time_s: <measured>
max_rss_bytes: <measured>
notes: same adapter and run_name layout as serial baseline
```

## How to interpret results

- If serial and parallel timings are similar, the grid is probably too small for multiprocessing to help.
- If wall time drops but max RSS rises sharply, record that tradeoff explicitly instead of only reporting the faster run.
- If repeated reruns vary significantly, compare the persisted `run_metadata.json` files first to confirm the grid and execution settings actually match.

For scientific-regression benchmarking of the astronomy workflow, keep using the reduced WASP-44 benchmark document; this Stage 5 guide is meant to make runtime and deployment comparisons repeatable, not to replace the science-facing regression baseline.
