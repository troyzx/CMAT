# CMAT: Companion Mass from TTV Modeling

[![DOI](https://zenodo.org/badge/777723832.svg)](https://zenodo.org/doi/10.5281/zenodo.13739646)

CMAT is a scientific Python project for constraining the upper mass of hidden planetary companions from transit timing variation (TTV) observations. In broader terms, it is a compact example of Bayesian inverse modeling for sparse, noisy time-series data using a physics-based forward simulator.

The current implementation fits TESS transit light curves, estimates per-transit center times with optimization and MCMC sampling, converts those estimates into TTV residuals, and evaluates grids of possible companion mass and period-ratio configurations with REBOUND N-body simulations. Candidate companions are rejected when simulated TTV amplitudes or goodness-of-fit statistics exceed the observed timing constraints; MEGNO simulations provide an additional dynamical-stability diagnostic.

## Project Scope

CMAT focuses on one astronomy use case:

- **Input:** TESS light curves, target metadata, and planetary system parameters.
- **Latent quantity:** the mass of an unseen companion across a grid of orbital period ratios.
- **Observation model:** transit timing residuals inferred from sparse photometric time series.
- **Forward model:** N-body integrations that generate synthetic transit timing variations.
- **Output:** upper companion-mass constraints and dynamical-stability maps.

This repository is research software. The code is useful as a worked scientific workflow and as a foundation for a future rebuild with stronger packaging, tests, reproducibility controls, and industry-facing examples.

## Core Workflow as Inverse Modeling

The astronomy vocabulary is specific, but the workflow shape is broader:

| CMAT stage | Current astronomy implementation | General inverse-modeling role |
| --- | --- | --- |
| Observation extraction | Fit light curves and recover per-transit center times | Turn raw observations into a compact signal representation |
| Uncertainty estimation | Propagate per-transit posterior uncertainty into `ttv_err` and timing residuals | Quantify how much ambiguity the downstream inference must absorb |
| Forward simulation | Run REBOUND over companion-mass and period-ratio grids | Generate synthetic observables from candidate latent states |
| Hypothesis rejection | Compare simulated TTVs with observed residuals using chi2/RMS or the experimental Bayesian mass-summary backend | Eliminate latent states that are inconsistent with the data |
| Decision-surface generation | Summarize surviving regimes as mass-limit curves plus MEGNO stability structure | Produce an interpretable operating surface rather than a single point estimate |

That framing is the main Stage 5 bridge: CMAT is still an astronomy codebase, but its workflow is also a concrete example of observation reduction, uncertainty-aware forward simulation, and constraint generation under a structured physical model.

## Repository Contents

- `cmat/` - Python source code for TESS light-curve fitting, transit-center inference, TTV construction, REBOUND simulations, mass-limit estimation, and MEGNO mapping.
- `data/WASP-44 b/` - example TESS and CSV data used by the included notebook.
- `docs/` - user-facing installation, quick-start, theory, API, data-format, and troubleshooting guides.
- `examples/` - small self-contained example scripts and notebooks, including both reduced astronomy workflows and a Stage 5 non-astronomy simulator-adapter example.
- `example.ipynb` - end-to-end example notebook for WASP-44 b.
- `requirements.txt` - runtime dependencies.
- `LICENSE` - GNU General Public License v3.0.

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Quick Start](docs/QUICKSTART.md)
- [Theory](docs/THEORY.md)
- [API Reference](docs/API_REFERENCE.md)
- [Data Formats](docs/DATA_FORMATS.md)
- [Benchmarking](docs/BENCHMARKING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Rebuild Notes](docs/rebuild/)

## Installation

Install the published package:

```bash
pip install CMAT-astro
```

Or run from a local checkout:

```bash
git clone https://github.com/troyzx/CMAT.git
cd CMAT
pip install -e . -c constraints.txt
```

CMAT is imported as:

```python
import cmat
```

For new code, prefer `cmat.TransitFitWorkflow` and `cmat.TTVSimulation`. The older `cmat.Fitlpf` and `cmat.ttv_sim` names remain available for compatibility.

The current industry rebuild also exposes import-safe pure helpers under `cmat.domain` for units, timing, residual scoring, and mass-limit extraction. Those functions are intended for new tests and adapters while the legacy notebook-compatible APIs remain unchanged.

## Example Notebook

Run the included full notebook:

```bash
jupyter notebook example.ipynb
```

The notebook uses the provided WASP-44 b data to demonstrate target setup, TESS data access, transit fitting, TTV residual construction, companion-mass grid simulation, and MEGNO visualization.

For a lighter interactive example that avoids remote data access and transit fitting, run:

```bash
jupyter notebook examples/synthetic_ttv_quickstart.ipynb
```

For the Stage 5 generic simulator-adapter path, run:

```bash
python examples/damped_oscillator_adapter.py
```

That example keeps the same inverse-modeling workflow shape but swaps the astronomy-specific simulator for a damped-oscillator forward model. It also writes a small portfolio bundle with a compact report, machine-readable table, summary JSON, and static figure under `artifacts/damped-oscillator-portfolio/`.

For a deployment-style variant with stable run folders and persisted run metadata, run:

```bash
python examples/damped_oscillator_batch.py --run-name damped-oscillator-baseline
```

That batch-style example writes `run_metadata.json` plus a nested portfolio bundle under `artifacts/deployment/<run-name>/`, which makes repeated reruns easier to compare or archive.

## Workflow: TESS Light Curves to Upper Mass Constraints

1. **Select a target.** Start with a known transiting planet and retrieve its system metadata, including stellar mass, planet mass, orbital period, radius, and reference transit time.
2. **Acquire TESS photometry.** Use MAST/TESS products or the cached example FITS files under `data/`.
3. **Fit the global light curve.** Build a TESS transit light-curve model with PyTransit and optimize the shared transit parameters.
4. **Fit individual transits.** Fit each transit segment separately and sample the posterior distribution for the transit center time.
5. **Construct the TTV series.** Convert posterior transit centers into epoch-indexed timing residuals and timing uncertainties.
6. **Define the companion grid.** Choose period-ratio values `P2/P1` and companion masses `M2` to evaluate.
7. **Run forward simulations.** Use REBOUND to simulate each candidate two-planet system and generate synthetic TTVs.
8. **Score candidate companions.** Compare simulated and inferred TTVs with `chi^2` and RMS criteria to estimate critical mass curves.
9. **Assess dynamical stability.** Run MEGNO calculations over the same grid and compare stability structure with the inferred mass limits.

## Transferable Workflow Pattern

Read the same workflow in a more general way:

1. **Extract observable structure** from raw measurements.
2. **Quantify uncertainty** on that reduced observable.
3. **Enumerate latent-state hypotheses** over a physically meaningful parameterization.
4. **Run a forward simulator** to predict what each latent state would look like in observation space.
5. **Reject inconsistent hypotheses** and summarize the remaining decision surface.

The astronomy-specific pieces are the transit-fitting front end and the REBOUND/MEGNO simulator, but the surrounding workflow is a reusable inverse-modeling pattern for sparse, noisy measurement problems.

## Transferable Engineering Skills

CMAT demonstrates engineering patterns that generalize beyond exoplanet research:

- **Bayesian inverse modeling:** estimate latent physical quantities from indirect, noisy observations.
- **Simulation-based inference:** use a forward simulator when closed-form likelihoods are limited or expensive.
- **Uncertainty quantification:** propagate posterior timing uncertainty into downstream constraint calculations.
- **Sparse time-series analysis:** infer signal structure from irregular, low-sample event timing data.
- **Physics-based forward simulation:** evaluate hypotheses with a domain simulator rather than a purely statistical surrogate.
- **Numerical workflow design:** combine optimization, MCMC sampling, grid search, multiprocessing, and diagnostic visualization.
- **Scientific reproducibility:** connect raw observational data, model assumptions, derived residuals, and final constraints in a traceable notebook workflow.

## Citation

If you use CMAT, cite the Zenodo archive:

- DOI: [10.5281/zenodo.13739646](https://doi.org/10.5281/zenodo.13739646)

Use the Zenodo record for version-specific citation metadata.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
