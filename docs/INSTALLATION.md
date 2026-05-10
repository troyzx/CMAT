# Installation

## Install from PyPI

```bash
pip install CMAT-astro
```

## Install from a local checkout

```bash
git clone https://github.com/troyzx/CMAT.git
cd CMAT
python -m venv .venv
source .venv/bin/activate
pip install -e . -c constraints.txt
```

## Import surface

CMAT is imported as:

```python
import cmat
```

Current top-level exports include:

- `cmat.Fitlpf`
- `cmat.ttv_sim`
- `cmat.TargetConfig`
- `cmat.FitControls`
- `cmat.SimulationGrid`
- `cmat.OutputConfig`
- `cmat.RunConfig`

## Environment notes

- Use `constraints.txt` for the current validated dependency stack.
- In restricted or sandboxed environments, set writable cache locations:

```bash
export MPLCONFIGDIR=/private/tmp/cmat-mplconfig
export XDG_CACHE_HOME=/private/tmp/cmat-cache
```

- `cmat.Fitlpf` still depends on the constrained PyTransit / Numba / llvmlite stack.
- `import cmat` and `cmat.ttv_sim` lazy-load cleanly without importing the full light-curve fitting stack.
