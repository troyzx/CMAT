# Stage 1 Packaging Baseline

Stage 1 starts by making packaging metadata explicit and tracked. The goal is to make local installs reproducible enough for validation before moving into source refactors.

## Changes Introduced

- Added `pyproject.toml` with setuptools build metadata.
- Moved package metadata into a tracked PEP 621 project table.
- Corrected the license classifier to GPL v3, matching `LICENSE`.
- Declared a conservative supported Python range: Python 3.10 and 3.11.
- Declared runtime dependencies in package metadata, including the constrained PyTransit, Numba, and llvmlite stack from Stage 0.
- Changed top-level package exports to lazy-load `Fitlpf` and `ttv_sim`, so importing `cmat` no longer imports the PyTransit light-curve stack.
- Updated the source-checkout install command to use editable package installation:

```bash
pip install -e . -c constraints.txt
```

## Deferred Packaging Work

- Convert `constraints.txt` into a more formal lock or environment strategy if needed.
- Add CI coverage across the supported Python matrix.
- Decide whether to keep `requirements.txt` as a compatibility file, generate it from package metadata, or remove it after downstream usage is understood.
- Add build and publish validation only after package metadata stabilizes.

## Local Artifact Policy

The repository now tracks a `.gitignore` for Python caches, notebook checkpoints, local build outputs, macOS files, and the legacy local `setup.py` file. The rebuild uses `pyproject.toml` as the tracked packaging entry point.

## Validation Snapshot

Commands run for this Stage 1 packaging slice:

```bash
python -m compileall cmat
python -m unittest discover -s tests
```

Result in the existing global environment: compile passes; `import cmat` succeeds; 6 tests pass and 1 TTV residual test is skipped because `cmat.Fitlpf` still reaches the incompatible global PyTransit import stack documented in Stage 0.

Commands run in the disposable constrained Python 3.11 environment:

```bash
env XDG_CACHE_HOME=/private/tmp/cmat-cache \
    MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
    python -m pip install -e . -c constraints.txt

env XDG_CACHE_HOME=/private/tmp/cmat-cache \
    MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
    python -c "import cmat; print(cmat.__all__)"

env XDG_CACHE_HOME=/private/tmp/cmat-cache \
    MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
    python -m unittest discover -v -s tests

env XDG_CACHE_HOME=/private/tmp/cmat-cache \
    MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
    python -m pip wheel --no-deps . -w /private/tmp/cmat-wheelhouse -c constraints.txt
```

Results: editable install succeeds, package import succeeds, 7 tests pass, and wheel build succeeds.
