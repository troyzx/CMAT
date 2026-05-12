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

- `cmat.TransitFitWorkflow` - preferred transit-fitting workflow name
- `cmat.TTVSimulation` - preferred forward-simulation workflow name
- `cmat.Fitlpf`
- `cmat.ttv_sim`
- `cmat.TargetConfig`
- `cmat.FitControls`
- `cmat.SimulationGrid`
- `cmat.ExecutionConfig`
- `cmat.BayesianScoringConfig`
- `cmat.ScoringConfig`
- `cmat.OutputConfig`
- `cmat.RunConfig`

The preferred public names are `cmat.TransitFitWorkflow` and `cmat.TTVSimulation`. The older `cmat.Fitlpf` and `cmat.ttv_sim` names remain available as compatibility aliases.

## Environment notes

- Use `constraints.txt` for the current validated dependency stack.
- In restricted or sandboxed environments, set writable cache locations:

```bash
export MPLCONFIGDIR=/private/tmp/cmat-mplconfig
export XDG_CACHE_HOME=/private/tmp/cmat-cache
```

- `cmat.TransitFitWorkflow` still depends on the constrained PyTransit / Numba / llvmlite stack.
- `import cmat`, `cmat.TTVSimulation`, and `cmat.ttv_sim` lazy-load cleanly without importing the full light-curve fitting stack.
