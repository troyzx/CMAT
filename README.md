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

## Repository Contents

- `cmat/` - Python source code for TESS light-curve fitting, transit-center inference, TTV construction, REBOUND simulations, mass-limit estimation, and MEGNO mapping.
- `data/WASP-44 b/` - example TESS and CSV data used by the included notebook.
- `docs/` - user-facing installation, quick-start, theory, API, data-format, and troubleshooting guides.
- `examples/` - small self-contained example scripts that avoid the full notebook workflow.
- `example.ipynb` - end-to-end example notebook for WASP-44 b.
- `requirements.txt` - runtime dependencies.
- `LICENSE` - GNU General Public License v3.0.

## Documentation

- [Installation](docs/INSTALLATION.md)
- [Quick Start](docs/QUICKSTART.md)
- [Theory](docs/THEORY.md)
- [API Reference](docs/API_REFERENCE.md)
- [Data Formats](docs/DATA_FORMATS.md)
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

## Example Notebook

Run the included notebook:

```bash
jupyter notebook example.ipynb
```

The notebook uses the provided WASP-44 b data to demonstrate target setup, TESS data access, transit fitting, TTV residual construction, companion-mass grid simulation, and MEGNO visualization.

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
