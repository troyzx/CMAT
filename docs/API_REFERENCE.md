# API Reference

CMAT's public surface is still evolving. The current practical entry points are:

## Top-level imports

```python
import cmat
```

- `cmat.Fitlpf` - transit-fitting workflow object
- `cmat.ttv_sim` - TTV and MEGNO forward-simulation workflow object
- `cmat.TargetConfig` - target metadata and data-root configuration
- `cmat.FitControls` - fitting iteration and sampler controls
- `cmat.SimulationGrid` - period-ratio, mass-grid, and MEGNO controls
- `cmat.OutputConfig` - artifact output paths
- `cmat.RunConfig` - composite workflow configuration

## Modules

### `cmat.base`

Contains the current `Fitlpf` implementation for:

- target parameter setup
- TESS light-curve fitting
- posterior transit-center extraction
- TTV residual construction

### `cmat.ttv_sim`

Contains the current forward-simulation workflow for:

- REBOUND TTV generation
- mass-threshold extraction
- MEGNO simulation
- plotting helpers

### `cmat.config`

Typed dataclasses for structured workflow inputs.

### `cmat.workflow`

Thin adapters from typed configuration objects to the existing workflow classes.

### `cmat.scoring`

Scoring helpers for comparing simulated and observed TTV residuals.

## Stability note

The configuration and workflow modules are the preferred boundary for future refactors. The lower-level fitting and simulation internals should still be treated as rebuild-era APIs rather than a finalized stable interface.
