# Quick Start

## Fastest validation path

The lightest validated example path currently uses the cached WASP-44 b data already stored in the repository.

Run the reduced smoke test:

```bash
MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
XDG_CACHE_HOME=/private/tmp/cmat-cache \
python -m unittest -v tests.test_notebook_smoke
```

This reduced path:

- loads cached timing residuals from `data/WASP-44 b/tc_data.csv`
- loads cached target properties from `data/WASP-44 b/prop_data.csv`
- runs a reduced TTV forward-simulation slice
- computes reduced mass-threshold outputs
- runs one reduced MEGNO point

In Stage 5 terms, that same reduced path can be read as:

1. **Observation extraction** - load the reduced timing residual series.
2. **Uncertainty estimation** - carry the per-epoch timing uncertainties in `ttv_err`.
3. **Forward simulation** - evaluate a small latent-state grid with REBOUND.
4. **Hypothesis rejection** - compare simulated and observed timing structure.
5. **Decision-surface generation** - summarize the result as reduced mass-limit and stability outputs.

## Minimal cached-data example

The reduced smoke path is built around the following minimal flow:

```python
import csv
from pathlib import Path

import numpy as np

from cmat import TTVSimulation

base = Path("data") / "WASP-44 b"

with (base / "tc_data.csv").open() as handle:
    ttv_rows = list(csv.DictReader(handle))
with (base / "prop_data.csv").open() as handle:
    prop_row = next(csv.DictReader(handle))

epochs = np.array([int(row["epochs"]) for row in ttv_rows])
ttv_mcmc = np.array([float(row["ttv_mcmc"]) for row in ttv_rows])
ttv_err = np.array([float(row["ttv_err"]) for row in ttv_rows])
target_properties = [
    {
        key: float(prop_row[key])
        for key in ("orbital_distance", "orbital_period", "Mp", "Ms", "Rs", "Rp")
    }
]
period_ratios = np.array([1.5])
companion_masses = np.array([10.0, 20.0])

simulation = TTVSimulation(
    epochs=epochs,
    ttv_mcmc=ttv_mcmc,
    ttv_err=ttv_err,
    rs=period_ratios,
    mp2s=companion_masses,
    prop=target_properties,
)
simulation.ttv_results = [
    simulation.calculate_rebound((1.5, 10.0)),
    simulation.calculate_rebound((1.5, 20.0)),
]
chi2_limit, rms_limit = simulation.get_critical_masses()
chi2_surface = simulation.get_chi2_surface()
```

This is not a replacement for the full notebook. It is the smallest currently maintained example of the repository's forward-simulation half.

For grids with at least two period ratios and two companion masses, the same scoring result can be visualized in the `P_2/P_1` by `M_2` plane:

```python
fig, ax = simulation.plot_chi2_contour(statistic="chi2")
fig, ax = simulation.plot_chi2_contour(statistic="log_likelihood")
```

## Full narrative example

For the current end-to-end astronomy workflow, run:

```bash
jupyter notebook example.ipynb
```

The notebook remains the full narrative example. It is heavier than the reduced smoke path and still includes steps that are intentionally outside the smallest validation loop.

## Synthetic example without remote data

For a smaller self-contained forward-simulation example, use either the script or the lightweight notebook:

```bash
python examples/synthetic_ttv_quickstart.py
```

```bash
jupyter notebook examples/synthetic_ttv_quickstart.ipynb
```

Both artifacts use the preferred `cmat.TTVSimulation` compatibility name while preserving the older `cmat.ttv_sim` path for existing code.

Both artifacts:

- define a small synthetic TTV series
- run a reduced companion grid
- compute reduced mass-threshold outputs
- evaluate one reduced MEGNO point

The script is the fastest non-notebook entry point. The notebook is the lightest interactive notebook path and is intentionally smaller than `example.ipynb`.

## What the quick start does not cover

The reduced path intentionally skips:

- remote data download
- global transit fitting
- per-transit posterior sampling
- full production TTV grids
- full MEGNO sweeps
