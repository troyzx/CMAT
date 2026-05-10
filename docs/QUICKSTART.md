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

## Minimal cached-data example

The reduced smoke path is built around the following minimal flow:

```python
import csv
from pathlib import Path

import numpy as np

from cmat.ttv_sim import ttv_sim

base = Path("data") / "WASP-44 b"

with (base / "tc_data.csv").open() as handle:
    ttv_rows = list(csv.DictReader(handle))
with (base / "prop_data.csv").open() as handle:
    prop_row = next(csv.DictReader(handle))

epochs = np.array([int(row["epochs"]) for row in ttv_rows])
ttv_mcmc = np.array([float(row["ttv_mcmc"]) for row in ttv_rows])
ttv_err = np.array([float(row["ttv_err"]) for row in ttv_rows])
prop = [
    {
        key: float(prop_row[key])
        for key in ("orbital_distance", "orbital_period", "Mp", "Ms", "Rs", "Rp")
    }
]

simulation = ttv_sim(
    epochs=epochs,
    ttv_mcmc=ttv_mcmc,
    ttv_err=ttv_err,
    rs=np.array([1.5]),
    mp2s=np.array([10.0, 20.0]),
    prop=prop,
)
simulation.ttv_results = [
    simulation.calculate_rebound((1.5, 10.0)),
    simulation.calculate_rebound((1.5, 20.0)),
]
chi2_limit, rms_limit = simulation.get_m_crit()
```

This is not a replacement for the full notebook. It is the smallest currently maintained example of the repository's forward-simulation half.

## Full narrative example

For the current end-to-end astronomy workflow, run:

```bash
jupyter notebook example.ipynb
```

The notebook remains the full narrative example. It is heavier than the reduced smoke path and still includes steps that are intentionally outside the smallest validation loop.

## Synthetic example without remote data

For a smaller self-contained forward-simulation example, run:

```bash
python examples/synthetic_ttv_quickstart.py
```

This script:

- defines a small synthetic TTV series
- runs a reduced companion grid
- computes reduced mass-threshold outputs
- evaluates one reduced MEGNO point

It is intentionally smaller than `example.ipynb` and does not require remote TESS access.

## What the quick start does not cover

The reduced path intentionally skips:

- remote data download
- global transit fitting
- per-transit posterior sampling
- full production TTV grids
- full MEGNO sweeps
