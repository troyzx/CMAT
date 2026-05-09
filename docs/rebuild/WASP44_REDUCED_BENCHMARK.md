# WASP-44 b Reduced Benchmark

This benchmark defines the smallest useful regression target for the rebuild. It is designed to protect the current workflow without requiring a full production TESS refit or a large REBOUND grid in every local test run.

## Purpose

- Preserve the current WASP-44 b example as the reference scientific workflow.
- Separate fast deterministic checks from slower simulation and notebook checks.
- Define expected artifacts before refactoring code paths that produce them.

## Inputs

- Target: `WASP-44 b`
- Timing residual file: `data/WASP-44 b/tc_data.csv`
- Metadata file: `data/WASP-44 b/prop_data.csv`
- Cached TESS products:
  - `tess2018263035959-s0003-0000000012862099-0123-s_lc.fits`
  - `tess2018263124740-s0003-s0003-0000000012862099-00405_dvt.fits`
  - `tess2018267104341-s0003-s0003-0000000012862099-00126_dvt.fits`

## Fast Baseline Invariants

These checks should run in normal unit tests:

- `tc_data.csv` has 8 timing rows.
- Epoch range is 1218 to 1226.
- Mean TTV residual is approximately 0 seconds.
- TTV residual sample standard deviation is approximately 95.55 seconds.
- Timing uncertainty range is approximately 85.45 to 154.19 seconds.
- Cached FITS file sizes match the tracked baseline.
- Synthetic scoring cases preserve current `chi^2`, RMS, and first-rejected-mass behavior.

## Reduced Simulation Benchmark

This tier is intended for a later Stage 0 or Stage 2 follow-up after dependency support is defined.

- Use cached WASP-44 b timing residuals instead of refitting the TESS light curve.
- Use a small period-ratio grid, for example 3 to 5 `P2/P1` values.
- Use a small companion-mass grid, for example 3 to 5 mass values.
- Run REBOUND TTV simulations with fixed configuration and record generated residuals.
- Compare mass-limit outputs with explicit tolerances.
- Save machine-readable outputs under a future ignored artifact directory such as `artifacts/stage0/`.

Expected future artifact names:

- `artifacts/stage0/wasp44_ttv_residuals.csv`
- `artifacts/stage0/wasp44_mass_constraints.csv`
- `artifacts/stage0/wasp44_megno_grid.csv`
- `artifacts/stage0/wasp44_benchmark_summary.json`

## Notebook Smoke Benchmark

This tier should run outside the fastest unit-test path:

- Execute a shortened notebook or script path using cached data only.
- Avoid remote MAST or ExoMAST calls.
- Use reduced optimizer, sampler, and grid settings.
- Verify that the workflow produces timing residual, mass-constraint, and plot artifacts.

## Tolerances

- Data-only invariants should be exact except for floating-point summaries.
- Timing residual summaries should use absolute tolerances at or below `1e-9` seconds where values are read from CSV.
- Simulation outputs should use looser domain-specific tolerances after the reduced REBOUND benchmark is recorded.
- Plot artifacts should be checked for existence and nonzero size, not pixel-perfect equality.
