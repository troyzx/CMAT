# Current Limitations

This file separates known limitations from the rebuild roadmap. It should be updated as limitations are resolved, confirmed as scientific scope, or converted into tracked implementation tasks.

## Environment and Packaging

- The repository now has tracked `pyproject.toml` metadata; `requirements.txt` still needs a compatibility decision after downstream usage is understood.
- `constraints.txt` provides a temporary rebuild-baseline guardrail for the high-risk import stack.
- Accessing `cmat.Fitlpf` fails in the observed global Python 3.11.14 environment because the PyTransit import path reaches an incompatible `numba` / `llvmlite` initialization path; the constrained disposable environment fixes that dependency pair.
- The package now lazy-loads top-level exports, so `import cmat` and `cmat.ttv_sim` no longer require the light-curve stack.
- Local ignored build artifacts exist in the workspace (`build/`, `dist/`, egg-info), but they are not part of the tracked source baseline.

## Data and Reproducibility

- The primary workflow is notebook-driven and does not yet save a formal provenance manifest.
- Typed configuration objects now represent target metadata, fitting controls, simulation grids, output paths, and random seeds; existing workflow code does not yet consume them or save a formal run record.
- The example relies on external astronomy services for fresh metadata or downloads when cached data are not used.
- The cached WASP-44 b data are useful for baseline tests, but they are not a general fixture strategy.

## Testing and Validation

- Stage 0 tests currently protect deterministic scoring helpers, TTV residual construction in the constrained environment, and cached WASP-44 b data invariants.
- The full transit-fitting workflow is not yet covered by automated tests.
- REBOUND simulation behavior is not yet protected by a reduced deterministic benchmark.
- MEGNO maps are not yet covered by automated regression checks.
- There is no CI configuration for tests, linting, or notebook smoke execution.

## Platform and Runtime

- Multiprocessing uses `fork`, which should be reviewed before claiming cross-platform support.
- Runtime and memory use are not benchmarked across grid sizes.
- Plotting imports can trigger Matplotlib cache behavior unless a writable `MPLCONFIGDIR` is configured.

## Scientific Scope

- Current constraints are framed for hidden companion mass limits from TTV observations, not full posterior characterization of all possible multiplanet architectures.
- Model assumptions, prior choices, and rejection thresholds need clearer documentation before broader reuse.
- Candidate rejection currently depends on grid resolution and selected scoring criteria; those choices should be exposed in benchmark metadata.
