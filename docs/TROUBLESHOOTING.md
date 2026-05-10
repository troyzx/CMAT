# Troubleshooting

## `cmat.Fitlpf` import fails with Numba / llvmlite errors

Use the constrained environment:

```bash
pip install -e . -c constraints.txt
```

The rebuild baseline depends on a specific PyTransit / Numba / llvmlite combination. The global environment may contain an incompatible pair even when `import cmat` itself succeeds.

## Which Python versions are currently validated?

The current CI baseline targets Python 3.10 and 3.11 with editable installation plus `constraints.txt`.

## Matplotlib or ArviZ cache errors

Set writable cache locations:

```bash
export MPLCONFIGDIR=/private/tmp/cmat-mplconfig
export XDG_CACHE_HOME=/private/tmp/cmat-cache
```

Use the same environment variables when running the quick-start example or the reduced notebook smoke test.

## The full notebook is too heavy for a quick check

Use the reduced smoke path instead:

```bash
MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
XDG_CACHE_HOME=/private/tmp/cmat-cache \
python -m unittest -v tests.test_notebook_smoke
```

Or run the small synthetic example:

```bash
MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
XDG_CACHE_HOME=/private/tmp/cmat-cache \
python examples/synthetic_ttv_quickstart.py
```

## Why do some reduced examples return empty mass-limit arrays?

An empty critical-mass array means the tested reduced companion grid did not cross the current rejection threshold. In the smallest smoke examples, that is expected: the grid is intentionally tiny and conservative because it is meant to validate plumbing, not to reproduce the full production result surface.

## Why is the example still notebook-driven?

The rebuild is staged. The current package now has typed configuration objects, workflow adapters, stronger validation, and a reduced smoke path, but the full transit-fitting workflow has not yet been decomposed into a complete reusable library pipeline.
