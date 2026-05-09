# Environment Baseline

This file records the current environment blocker and the repository-level mitigation added before deeper rebuild work.

## Observed Import Failure

The local Python 3.11.14 environment fails on:

```bash
python -c "import cmat; print(cmat.__all__)"
```

The failure occurs while `cmat.base` imports PyTransit, which imports Numba and llvmlite:

```text
RuntimeError: llvmlite.binding.initialize() is deprecated and will be removed.
LLVM initialization is now handled automatically.
```

## Root Cause Found Locally

The active environment has an invalid dependency pair:

```text
numba 0.61.2
llvmlite 0.46.0
```

Numba's installed package metadata requires:

```text
llvmlite >=0.44,<0.45
```

`python -m pip check` also reports this mismatch directly.

## Repository Mitigation

`constraints.txt` now pins the high-risk import stack used by the rebuild baseline:

```bash
pip install -r requirements.txt -c constraints.txt
```

The constraints file is not a final lockfile. It is a narrow guardrail until Stage 1 moves package metadata to `pyproject.toml`, defines supported Python versions, and establishes a tested dependency matrix.

## Disposable Environment Validation

A disposable Python 3.11 virtual environment was created under `/private/tmp/cmat-rebuild-env` and installed with:

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

The constrained environment resolved the invalid Numba/llvmlite pair:

```text
numba 0.61.2
llvmlite 0.44.0
numpy 2.2.6
scipy 1.15.3
```

`python -m pip check` reported:

```text
No broken requirements found.
```

In the Codex sandbox, `import cmat` also requires writable cache directories for Matplotlib and ArviZ:

```bash
env XDG_CACHE_HOME=/private/tmp/cmat-cache \
    MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
    python -c "import cmat; print(cmat.__all__)"
```

Result:

```text
['Fitlpf', 'ttv_sim']
```

The same disposable environment passed:

```bash
env XDG_CACHE_HOME=/private/tmp/cmat-cache \
    MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
    python -m unittest discover -s tests
```

Result: 5 tests passed.

## Current Recommendation

- Do not modify the global Anaconda environment as part of the rebuild.
- Use a fresh virtual environment for validation.
- Install source-checkout dependencies with `requirements.txt` plus `constraints.txt`.
- Set `XDG_CACHE_HOME` and `MPLCONFIGDIR` to writable paths in sandboxed or restricted environments.
- Treat any remaining import failures as Stage 1 packaging/environment issues before expanding notebook or end-to-end workflow tests.
