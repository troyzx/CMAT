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

## Preferred boundary

For new code, prefer the configuration and workflow modules over reaching directly into notebook-era internals.

Example:

```python
import numpy as np

from cmat.config import RunConfig, SimulationGrid, TargetConfig
from cmat.workflow import make_ttv_simulation

config = RunConfig(
    target=TargetConfig("WASP-44 b"),
    simulation=SimulationGrid(
        period_ratios=[1.5, 2.0],
        companion_masses=[10.0, 20.0],
        n_transit_simulations=6,
        megno_dt=0.1,
        megno_runtime=100.0,
    ),
)

simulation = make_ttv_simulation(
    config,
    epochs=np.array([0, 1, 2]),
    ttv_mcmc=np.array([0.0, 1.0, 0.0]),
    ttv_err=np.ones(3),
    prop=[{"orbital_distance": 1.0, "orbital_period": 1.0, "Mp": 1.0, "Ms": 1.0}],
)
```

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

- `TargetConfig` - target name, local data root, product subgroup selection
- `FitControls` - optimizer and sampler counts
- `SimulationGrid` - period ratios, companion masses, and MEGNO controls
- `OutputConfig` - output directories and metadata destinations
- `RunConfig` - composite run configuration

### `cmat.workflow`

Thin adapters from typed configuration objects to the existing workflow classes.

- `legacy_data_dir(target)` - normalize a `TargetConfig` to the current `Fitlpf` data-root string
- `make_fit_lpf(target)` - build a legacy `Fitlpf` object from a typed target config
- `make_ttv_simulation(config, ...)` - build a `ttv_sim` object from typed simulation settings and TTV arrays
- `workflow_manifest(config, ...)` - build a JSON-serializable run manifest

### `cmat.scoring`

Scoring helpers for comparing simulated and observed TTV residuals.

- `get_chi2(...)` - minimum aligned `chi^2` score over discrete epoch offsets
- `get_rms(...)` - root-mean-square amplitude of a simulated TTV series

## Stability note

The configuration and workflow modules are the preferred boundary for future refactors. The lower-level fitting and simulation internals should still be treated as rebuild-era APIs rather than a finalized stable interface.
