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

## Full narrative example

For the current end-to-end astronomy workflow, run:

```bash
jupyter notebook example.ipynb
```

The notebook remains the full narrative example. It is heavier than the reduced smoke path and still includes steps that are intentionally outside the smallest validation loop.
