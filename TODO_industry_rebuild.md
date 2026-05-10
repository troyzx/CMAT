# CMAT to Industry Rebuild Roadmap

This roadmap keeps the current scientific objective intact: constrain hidden companion masses from TTV observations using transit fitting and physics-based forward simulation. The staged plan focuses on refactoring, validation, packaging, examples, and broader inverse-modeling presentation.

## Stage 0: Baseline and Scope Control

- Freeze the current scientific behavior with a reproducible WASP-44 b benchmark run.
- Record expected notebook artifacts: inferred transit centers, TTV residuals, critical mass curves, and MEGNO map structure.
- Inventory public APIs, file inputs, data products, units, and implicit assumptions.
- Define non-goals for the first rebuild: no algorithmic changes until regression tests protect current outputs.
- Audit license and package metadata for consistency before a new release.

## Stage 1: Packaging and Project Structure

- Move packaging metadata to `pyproject.toml` with explicit Python version support and dependency bounds. _(Started in Stage 1 packaging branch.)_
- Separate modules by responsibility: data access, light-curve fitting, posterior extraction, TTV construction, forward simulation, scoring, plotting, and examples. _(Started by extracting scoring helpers from forward simulation.)_
- Add typed configuration objects for target metadata, fitting controls, simulation grids, and output paths. _(Started in Stage 1 project-structure slice.)_
- Replace notebook-only workflow state with reusable library functions and optional command-line entry points. _(Started with configuration-to-workflow adapters; CLI deferred.)_
- Add structured logging and deterministic random seed handling where supported by dependencies.

## Stage 2: Validation and Tests

- Add unit tests for unit conversions, epoch calculation, TTV residual construction, `chi^2` scoring, RMS scoring, and mass-threshold extraction.
- Add small synthetic-system tests with known injected timing offsets and known recovery expectations.
- Use inject-recovery comparisons to evaluate future scoring backends before replacing the current `chi^2` / RMS baseline.
- Add regression tests for REBOUND simulation outputs using short, deterministic runs.
- Add notebook smoke tests that execute a reduced example without requiring a full production grid. Use the cached WASP-44 b data path, skip the remote download cell, reduce the TTV simulation to a minimal deterministic fixture, and leave full-grid and MEGNO sweeps to slower manual validation until a smaller example exists.
- Configure continuous integration for linting, tests, and notebook execution on a supported Python matrix. _(Started with constrained editable-install validation, `compileall`, and `unittest` on Python 3.10/3.11; notebook smoke execution is still deferred.)_

## Stage 3: Documentation and Examples

- Split the documentation into installation, quick start, theory, API reference, data formats, and troubleshooting.
- Keep `example.ipynb` as the narrative astronomy workflow and add a smaller quick-start notebook for fast validation.
- Add a synthetic-data example that does not require remote TESS queries.
- Document input schemas for FITS products, transit-center CSV files, and planetary metadata.
- Add interpretation guidance for mass-limit curves, MEGNO maps, unstable configurations, and observational caveats.

## Stage 4: Inference and Performance Refactor

- Make the probabilistic model explicit: priors, fitted parameters, posterior samples, likelihood assumptions, and downstream uncertainty propagation.
- Introduce a single-target Bayesian TTV scoring backend only in this stage: start with a nuisance-parameter likelihood over epoch shift, constant offset, and extra jitter, keep it parallel to the current `chi^2` / RMS scoring, and use Stage 2 inject-recovery results to judge whether it should become the default.
- Provide clean interfaces for alternative samplers, reduced-order simulators, or approximate Bayesian computation experiments.
- Cache expensive intermediate products, including downloaded light curves, posterior samples, simulated TTV grids, and MEGNO grids.
- Improve parallel execution controls for local workstations and batch environments.
- Add provenance metadata to every saved result: code version, dependency versions, target parameters, grid definition, random seeds, and runtime settings.

## Stage 5: Industry-Facing Extensions

- Present the core workflow as a general inverse-modeling pattern: observation extraction, uncertainty estimation, forward simulation, hypothesis rejection, and decision surface generation.
- Add a simulator adapter interface so the same workflow can be demonstrated with non-astronomy physical systems.
- Add example outputs suitable for technical portfolios: compact reports, static figures, and machine-readable result tables.
- Provide deployment-oriented examples for batch execution, artifact storage, and reproducible reruns.
- Add benchmarking notes covering runtime, grid size, multiprocessing strategy, and memory use.

## Stage 6: Release and Maintenance

- Add `CITATION.cff`, contributor guidance, changelog, and release checklist.
- Establish semantic versioning and versioned documentation.
- Add pre-commit checks for formatting, linting, import sorting, and documentation link validation.
- Publish a documented release artifact with validated examples and archived benchmark outputs.
- Track known limitations separately from future research ideas so users can distinguish software debt from scientific scope.
