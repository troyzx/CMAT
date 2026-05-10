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
- Typed configuration objects and thin workflow adapters now represent target metadata, fitting controls, simulation grids, output paths, and random seeds; the full notebook workflow is not yet decomposed into reusable library steps or saved as a formal run record.
- The example relies on external astronomy services for fresh metadata or downloads when cached data are not used.
- The cached WASP-44 b data are useful for baseline tests, but they are not a general fixture strategy.

## Testing and Validation

- Stage 0 tests currently protect deterministic scoring helpers, TTV residual construction in the constrained environment, and cached WASP-44 b data invariants.
- The full transit-fitting workflow is not yet covered by automated tests.
- A reduced deterministic REBOUND transit-timing fixture now protects one minimal simulation path, but broader instability and grid-behavior coverage is still missing.
- Notebook smoke execution is still missing; the current reduced plan is to reuse cached WASP-44 b data, skip remote download calls, and clamp the notebook path to minimal TTV simulation coverage instead of the production grid and MEGNO sweeps.
- MEGNO maps are not yet covered by automated regression checks.
- CI now covers constrained editable installs, `compileall`, and the `unittest` suite on Python 3.10 and 3.11, but notebook smoke execution and linting are still not automated.

## Platform and Runtime

- Multiprocessing uses `fork`, which should be reviewed before claiming cross-platform support.
- Runtime and memory use are not benchmarked across grid sizes.
- Plotting imports can trigger Matplotlib cache behavior unless a writable `MPLCONFIGDIR` is configured.

## Scientific Scope

- Current constraints are framed for hidden companion mass limits from TTV observations, not full posterior characterization of all possible multiplanet architectures.
- Model assumptions, prior choices, and rejection thresholds need clearer documentation before broader reuse.
- Candidate rejection currently depends on grid resolution and selected scoring criteria; those choices should be exposed in benchmark metadata.
