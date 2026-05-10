# Stage 1 Project Structure: Configuration Boundary

This Stage 1 slice introduces typed configuration objects before changing the
workflow implementation. The goal is to make target inputs, fitting controls,
simulation grids, output paths, and random seeds explicit without changing the
current scientific algorithms.

## Changes Introduced

- Added `cmat.config.TargetConfig` for target names, local data roots, and TESS product subgroup selection.
- Added `cmat.config.FitControls` for optimizer and sampler counts currently embedded in fitting methods.
- Added `cmat.config.SimulationGrid` for companion period-ratio grids, companion-mass grids, transit simulation count, and MEGNO controls.
- Added `cmat.config.OutputConfig` for run artifact directories, table directories, figure directories, and metadata paths.
- Added `cmat.config.RunConfig` as a composite object for future notebook, command-line, and batch workflows.
- Exposed the configuration objects through lazy top-level `cmat` exports without importing the light-curve stack.

## Deliberate Non-Changes

- Existing transit fitting, posterior extraction, TTV residual construction, REBOUND simulation, scoring, and plotting code paths are unchanged.
- Existing default iteration counts and simulation defaults are preserved in the configuration objects but are not yet wired into `Fitlpf` or `ttv_sim`.
- Configuration serialization is limited to JSON-compatible dictionaries; no new YAML, TOML, or locking dependency is introduced.

## Integration Plan

1. Add workflow helper functions that accept these configuration objects while delegating to the current implementation.
2. Preserve current defaults when adapting `Fitlpf` and `ttv_sim` so regression tests can compare behavior before and after integration.
3. Add provenance writing after the reduced workflow can run from configuration objects end to end.
4. Add optional command-line entry points only after configuration-driven library functions exist.

## Validation

New tests cover path normalization, parameter validation, simulation-grid parameter ordering, top-level lazy exports, and JSON-serializable run configuration dictionaries.
