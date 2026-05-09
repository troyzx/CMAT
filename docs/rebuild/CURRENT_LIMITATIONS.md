# Current Limitations

This file separates known limitations from the rebuild roadmap. It should be updated as limitations are resolved, confirmed as scientific scope, or converted into tracked implementation tasks.

## Environment and Packaging

- The repository currently has no tracked `pyproject.toml`, `setup.cfg`, or `setup.py`; packaging metadata needs to be made explicit in Stage 1.
- Dependencies are listed without version bounds in `requirements.txt`; `constraints.txt` provides a temporary rebuild-baseline guardrail for the high-risk import stack.
- `import cmat` fails in the observed local Python 3.11.14 environment because the PyTransit import path reaches an incompatible `numba` / `llvmlite` initialization path.
- The current package import is eager: importing `cmat` imports the light-curve stack even when the user only needs `ttv_sim`.
- Local ignored build artifacts exist in the workspace (`build/`, `dist/`, egg-info), but they are not part of the tracked source baseline.

## Data and Reproducibility

- The primary workflow is notebook-driven and does not yet save a formal provenance manifest.
- MCMC settings, optimizer settings, random seeds, dependency versions, and grid definitions are not captured in a machine-readable run record.
- The example relies on external astronomy services for fresh metadata or downloads when cached data are not used.
- The cached WASP-44 b data are useful for baseline tests, but they are not a general fixture strategy.

## Testing and Validation

- Stage 0 tests currently protect deterministic scoring helpers and cached WASP-44 b data invariants.
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
