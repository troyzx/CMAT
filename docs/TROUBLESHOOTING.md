# Troubleshooting

## `cmat.Fitlpf` import fails with Numba / llvmlite errors

Use the constrained environment:

```bash
pip install -e . -c constraints.txt
```

The rebuild baseline depends on a specific PyTransit / Numba / llvmlite combination. The global environment may contain an incompatible pair even when `import cmat` itself succeeds.

## Matplotlib or ArviZ cache errors

Set writable cache locations:

```bash
export MPLCONFIGDIR=/private/tmp/cmat-mplconfig
export XDG_CACHE_HOME=/private/tmp/cmat-cache
```

## The full notebook is too heavy for a quick check

Use the reduced smoke path instead:

```bash
python -m unittest -v tests.test_notebook_smoke
```

## Why is the example still notebook-driven?

The rebuild is staged. The current package now has typed configuration objects, workflow adapters, stronger validation, and a reduced smoke path, but the full transit-fitting workflow has not yet been decomposed into a complete reusable library pipeline.
