# Stage 0 Baseline and Scope Control

This document freezes the current repository state before the CMAT to Industry rebuild. It is intentionally descriptive: the goal is to preserve scientific behavior before changing package structure, dependency management, or algorithms.

## Reference State

- Branch: `codex/industry-rebuild-stage-0`
- Base commit: `8a21507` (`Merge pull request #2 from troyzx/cmat-industry-rebuild`)
- Scope of this stage: inventory, validation, risk capture, and benchmark definition.
- Non-goal: changing transit fitting, TTV construction, N-body simulation, mass-threshold logic, or MEGNO behavior.

## Current Public Workflow

The main user-facing workflow is notebook-driven:

1. Create a `cmat.Fitlpf` object for a target planet.
2. Retrieve target metadata from ExoMAST.
3. Download or reuse TESS light-curve products.
4. Run global light-curve optimization with PyTransit.
5. Fit individual transits and sample transit-center posteriors.
6. Convert fitted transit centers into TTV residuals and uncertainties.
7. Build period-ratio and companion-mass grids.
8. Run REBOUND simulations over the grid.
9. Estimate critical companion-mass curves using `chi^2` and RMS criteria.
10. Run MEGNO simulations to map dynamical stability.

## API Inventory

Tracked modules:

- `cmat.__init__`
  - Exports `Fitlpf` and `ttv_sim`.
- `cmat.base`
  - `Fitlpf`: target metadata lookup, TESS download, global transit fitting, single-transit fitting, posterior extraction, TTV residual calculation, and plotting.
  - Helper functions for ExoMAST lookup, uncertainty extraction, colormap truncation, and data saving.
- `cmat.singlefit`
  - `SingleFit`: PyTransit `BaseLPF` subclass with plotting helpers for single-transit diagnostics and posterior corner plots.
- `cmat.ttv_sim`
  - `ttv_sim`: REBOUND-based TTV simulation, mass-threshold extraction, MEGNO simulation, and MEGNO plotting.
  - `get_chi2` and `get_rms`: scoring helpers used by the grid workflow.
- `cmat.constant` and `cmat.utils`
  - Duplicate or overlapping constants and helper utilities that should be consolidated after baseline protection exists.

## Data Inventory

Tracked example target: `WASP-44 b`.

- `data/WASP-44 b/prop_data.csv`
  - 2 catalog rows plus header.
  - Contains stellar, planetary, orbital, and transit metadata from exoplanet catalog sources.
- `data/WASP-44 b/tc_data.csv`
  - 8 inferred timing-residual rows plus header.
  - Epoch range: 1218 to 1226.
  - TTV residual mean: approximately 0 seconds.
  - TTV residual sample standard deviation: approximately 95.55 seconds.
  - Timing uncertainty range: approximately 85.45 to 154.19 seconds.
- `data/WASP-44 b/mastDownload/...`
  - 1 TESS light-curve FITS file: 1,998,720 bytes.
  - 2 TESS DVT FITS files: 3,968,640 bytes and 2,949,120 bytes.

## Current Validation Snapshot

Commands run on the Stage 0 branch:

```bash
python -m compileall cmat
```

Result: passed for all tracked Python modules.

```bash
python -m unittest discover -s tests
```

Result: passed 5 tests covering deterministic scoring helpers, first-rejected-mass extraction, and cached WASP-44 b data invariants. The scoring tests load `cmat/ttv_sim.py` directly to avoid the package-level PyTransit import boundary described below.

Additional Stage 0 files:

- `docs/rebuild/CURRENT_LIMITATIONS.md` records known limitations separately from implementation tasks.
- `docs/rebuild/ENVIRONMENT_BASELINE.md` records the current import failure and dependency-constraint mitigation.
- `docs/rebuild/WASP44_REDUCED_BENCHMARK.md` defines the reduced WASP-44 b benchmark contract.

```bash
python -c "import cmat; print(cmat.__all__)"
```

Result: failed in the local Python 3.11.14 environment while importing `pytransit` through `cmat.base`. The observed failure originates in the `numba` / `llvmlite` stack:

```text
RuntimeError: llvmlite.binding.initialize() is deprecated and will be removed.
LLVM initialization is now handled automatically.
```

This should be treated as an environment and dependency-compatibility baseline issue. Do not patch scientific code to work around it until dependency versions and supported Python versions are made explicit.

Observed local dependency versions:

- Python: 3.11.14
- PyTransit: 2.6.14
- numba: 0.61.2
- llvmlite: 0.46.0
- rebound: 3.26.3
- emcee: 3.1.4

Environment mitigation added after this finding:

- `constraints.txt` constrains the high-risk PyTransit/Numba/llvmlite stack.
- Source checkout installation now uses `pip install -r requirements.txt -c constraints.txt`.
- A disposable constrained Python 3.11 environment successfully imported `cmat` when `XDG_CACHE_HOME` and `MPLCONFIGDIR` pointed to writable directories.

## Known Risks Before Refactor

- Before this branch, no tracked automated tests or CI configuration protected behavior.
- The first Stage 0 tests cover deterministic scoring helpers and cached WASP-44 b data invariants; there is still no CI, and the tests do not yet validate the full TESS, MCMC, REBOUND-grid, or MEGNO workflow.
- Runtime dependencies are listed without version bounds in `requirements.txt`; `constraints.txt` is a temporary mitigation, not final package metadata.
- There is no tracked `pyproject.toml`, `setup.cfg`, or `setup.py` packaging metadata in the repository state.
- Importing `cmat` eagerly imports the light-curve stack, so lightweight use of `ttv_sim` is coupled to PyTransit availability.
- External data access depends on ExoMAST and MAST availability.
- The notebook is the primary executable workflow and contains generated output state.
- MCMC and optimizer settings are not captured in a reproducibility manifest.
- Multiprocessing uses `fork`, which should be reviewed for cross-platform behavior.
- Unit conversions and physical units are represented as module-level constants in more than one place.

## Stage 0 Exit Criteria

Before moving into structural refactoring:

- Add deterministic tests for TTV residual construction once the environment/import path is stable.
- Implement the reduced WASP-44 b benchmark configuration with expected artifact names and tolerances.
- Decide supported Python versions and pin or bound high-risk scientific dependencies.
- Record current limitations separately from rebuild tasks.
