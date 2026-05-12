# API Reference

CMAT's public surface is still evolving. The current practical entry points are:

## Top-level imports

```python
import cmat
```

- `cmat.TransitFitWorkflow` - preferred transit-fitting workflow name
- `cmat.TTVSimulation` - preferred TTV and MEGNO forward-simulation workflow name
- `cmat.Fitlpf` - legacy-compatible alias for `cmat.TransitFitWorkflow`
- `cmat.ttv_sim` - legacy-compatible alias for `cmat.TTVSimulation`
- `cmat.TargetConfig` - target metadata and data-root configuration
- `cmat.FitControls` - fitting iteration and sampler controls
- `cmat.SimulationGrid` - period-ratio, mass-grid, and MEGNO controls
- `cmat.BayesianScoringConfig` - typed controls for the Bayesian nuisance-parameter scorer
- `cmat.ScoringConfig` - typed selector for the current scoring backend
- `cmat.OutputConfig` - artifact output paths
- `cmat.RunConfig` - composite workflow configuration

## Preferred boundary

For new code, prefer the configuration and workflow modules over reaching directly into notebook-era internals. That boundary is where the rebuild is making inputs explicit and JSON-serializable without changing the scientific implementation underneath.

Example:

```python
import numpy as np

from cmat.config import OutputConfig, RunConfig, ScoringConfig, SimulationGrid, TargetConfig
from cmat.workflow import make_ttv_simulation, workflow_manifest

config = RunConfig(
    target=TargetConfig("WASP-44 b"),
    simulation=SimulationGrid(
        period_ratios=[1.5, 2.0],
        companion_masses=[10.0, 20.0],
        n_transit_simulations=6,
        megno_dt=0.1,
        megno_runtime=100.0,
    ),
    scoring=ScoringConfig(backend="chi2_rms"),
    output=OutputConfig(root_dir="artifacts", run_name="wasp44b-reduced"),
    random_seed=42,
)

simulation = make_ttv_simulation(
    config,
    epochs=np.array([0, 1, 2]),
    ttv_mcmc=np.array([0.0, 1.0, 0.0]),
    ttv_err=np.ones(3),
    prop=[{"orbital_distance": 1.0, "orbital_period": 1.0, "Mp": 1.0, "Ms": 1.0, "Rs": 1.0, "Rp": 1.0}],
)
manifest = workflow_manifest(config, dependency_versions={"numpy": "2.x"})
```

## Configuration objects

### `TargetConfig`

Target-level inputs for data discovery and local storage.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `planet_name` | `str` | required | Must be a non-empty string |
| `data_dir` | `Path \| str` | `Path("data")` | Root directory containing per-target folders |
| `product_subgroups` | iterable of `str` | `("LC",)` | Must contain at least one non-empty value |

Derived property:

- `target_data_dir` - `data_dir / planet_name`

Serialization helper:

- `to_dict()` - returns a JSON-friendly target description

### `FitControls`

Optimizer and sampler controls for the light-curve fitting stages.

| Field | Default |
| --- | --- |
| `global_niter` | `200` |
| `global_npop` | `30` |
| `single_niter` | `100` |
| `single_npop` | `50` |
| `mcmc_steps` | `2500` |
| `mcmc_thin` | `25` |
| `mcmc_repeats` | `4` |

All fields must be positive integers.

### `SimulationGrid`

Forward-simulation grid for companion period ratios and masses.

| Field | Type | Default | Validation |
| --- | --- | --- | --- |
| `period_ratios` | 1D iterable of float | required | finite, positive, non-empty |
| `companion_masses` | 1D iterable of float | required | finite, positive, non-empty, strictly increasing |
| `n_transit_simulations` | `int` | `80` | positive integer |
| `megno_dt` | `float` | `1/20` | finite, positive |
| `megno_runtime` | `float` | `1e4` | finite, positive |

Key helpers:

- `parameter_count` - total number of `(period_ratio, companion_mass)` pairs
- `parameter_pairs()` - list of parameter tuples in the same mass-major ordering used by the current simulation loops
- `to_dict()` - JSON-friendly grid description

### `OutputConfig`

Filesystem destinations for run artifacts.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `root_dir` | `Path \| str` | `Path("artifacts")` | Base artifact directory |
| `run_name` | `str \| None` | `None` | Optional per-run subdirectory |

Derived properties:

- `run_dir` - `root_dir` or `root_dir / run_name`
- `tables_dir` - `run_dir / "tables"`
- `figures_dir` - `run_dir / "figures"`
- `metadata_path` - `run_dir / "run_metadata.json"`

### `ScoringConfig`

Typed selector for the current mass-threshold scoring backend.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `backend` | `str` | `"chi2_rms"` | Supported values: `"chi2_rms"` and `"bayesian"` |
| `bayesian` | `BayesianScoringConfig \| None` | `None` | Auto-filled when `backend="bayesian"`; rejected otherwise |

### `BayesianScoringConfig`

Typed controls for the Stage 4 Bayesian scorer.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `credible_interval` | `float` | `0.997` | Must be strictly between 0 and 1 |
| `posterior_sample_count` | `int` | `2000` | Retained posterior draws from the nuisance-parameter sampler; also reused by evidence-backed summary outputs |
| `warmup_draws` | `int` | `1000` | Sampler warmup steps discarded before summarizing the posterior |
| `nuisance_parameters` | tuple of `str` | `("epoch_shift", "baseline_offset", "jitter")` | Supported nuisance parameters for the single-target Bayesian TTV likelihood |
| `store_chains` | `bool` | `False` | Include retained posterior samples in the JSON-ready scoring summary |

The default path remains `chi2_rms`, while the Bayesian branch now runs an `emcee`-based nuisance-parameter sampler for posterior summaries and derives probability-style mass-limit outputs from a marginal-likelihood evidence approximation.

### `RunConfig`

Composite configuration for fitting and simulation runs.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `target` | `TargetConfig` | required | mandatory |
| `fit` | `FitControls` | default factory | fitting controls |
| `simulation` | `SimulationGrid \| None` | `None` | required by `make_ttv_simulation(...)` |
| `scoring` | `ScoringConfig` | default factory | typed scoring backend selector |
| `output` | `OutputConfig` | default factory | artifact destinations |
| `random_seed` | `int \| None` | `None` | must be non-negative if provided |

`RunConfig.to_dict()` returns a fully JSON-serializable nested configuration object.

## Workflow adapters

The `cmat.workflow` module provides the current rebuild boundary around the legacy classes.

| Function | Returns | Purpose |
| --- | --- | --- |
| `legacy_data_dir(target)` | `str` | Normalizes `TargetConfig.data_dir` to the trailing-slash string expected by the fitting workflow |
| `make_fit_lpf(target)` | `TransitFitWorkflow` | Builds a transit-fitting object from a typed target config |
| `make_ttv_simulation(config, *, epochs, ttv_mcmc, ttv_err, prop, scoring_backend=None)` | `TTVSimulation` | Builds a forward-simulation object from typed config plus observed TTV arrays and selects the configured default scorer |
| `workflow_manifest(config, *, dependency_versions=None, notes=None, scoring_summary=None)` | `dict` | Builds a JSON-serializable run manifest |

### `make_ttv_simulation(...)` input contract

`make_ttv_simulation(...)` currently expects:

- `config.simulation` to be present
- `epochs`, `ttv_mcmc`, and `ttv_err` to be aligned one-dimensional arrays
- `prop` to be a list whose first item contains at least:
  - `orbital_distance`
  - `orbital_period`
  - `Mp`
  - `Ms`
  - `Rs`
  - `Rp`
- `scoring_backend`, if provided, to expose a `critical_masses(...)` method compatible with the current mass-threshold extraction flow

The adapter forwards:

- `SimulationGrid.period_ratios` -> `TTVSimulation.rs`
- `SimulationGrid.companion_masses` -> `TTVSimulation.mp2s`
- `SimulationGrid.n_transit_simulations` -> `TTVSimulation.N`
- `ScoringConfig.backend` -> default scorer created for `TTVSimulation.scoring_backend`
- `SimulationGrid.megno_dt` / `megno_runtime` -> mutable MEGNO controls on the returned object

`workflow_manifest(...)` also accepts `scoring_summary`, which can be either a plain dict or a `MassThresholds` object. This is the current provenance-friendly path for carrying scoring metadata into a JSON-serializable run record.

## Scoring helpers

The `cmat.scoring` module now contains both the current comparison helpers and the first explicit backend boundary for Stage 4 scoring work.

| Function | Purpose |
| --- | --- |
| `get_chi2(ttv_rebound, epoch, ttv_mcmc, ttv_err)` | Minimum aligned `chi^2` score over discrete epoch offsets |
| `get_rms(ttv_rebound)` | Root-mean-square amplitude of a simulated TTV series |
| `get_chi2_v(...)` | Vectorized form of `get_chi2` used across simulation grids |
| `get_rms_v(...)` | Vectorized form of `get_rms` used across simulation grids |

| Interface | Purpose |
| --- | --- |
| `MassThresholds` | Dataclass holding the current `chi2` / RMS critical-mass curves plus backend metadata |
| `MassThresholdScorer` | Protocol for backend objects that expose `critical_masses(...)` |
| `Chi2AndRmsMassThresholdScorer` | Default backend that preserves the current legacy `chi^2` / RMS behavior |
| `BayesianMassThresholdScorer` | Bayesian backend that samples epoch shift, baseline offset, and extra jitter, then summarizes posterior mass support |
| `supported_mass_threshold_backends()` | Return the backend names currently accepted by `ScoringConfig.backend` |

Minimal custom-backend example:

```python
import numpy as np

from cmat import TTVSimulation
from cmat.scoring import MassThresholds


class FixedDemoScorer:
    def critical_masses(
        self,
        *,
        ttv_results,
        epoch,
        ttv_mcmc,
        ttv_err,
        period_ratios,
        companion_masses,
    ) -> MassThresholds:
        return MassThresholds(
            chi2=np.full(len(period_ratios), 12.0),
            rms=np.full(len(period_ratios), 18.0),
            backend="fixed-demo",
        )


simulation = TTVSimulation(
    epochs=np.array([0, 1, 2]),
    ttv_mcmc=np.array([0.0, 1.0, 0.0]),
    ttv_err=np.ones(3),
    rs=np.array([1.5, 2.0]),
    mp2s=np.array([10.0, 20.0]),
    prop=[{"orbital_distance": 1.0, "orbital_period": 1.0, "Mp": 1.0, "Ms": 1.0, "Rs": 1.0, "Rp": 1.0}],
    scoring_backend=FixedDemoScorer(),
)
simulation.ttv_results = [np.zeros(4)] * 4
summary = simulation.get_critical_masses(), simulation.get_scoring_summary()
```

For a runnable version, see [`examples/custom_scoring_backend.py`](../examples/custom_scoring_backend.py).

## Workflow classes

The classes below are still rebuild-era APIs. The preferred public names are clearer aliases layered on top of the same current implementations.

### `cmat.TransitFitWorkflow`

Legacy alias: `cmat.Fitlpf`

Constructor:

```python
TransitFitWorkflow(planet_name: str, datadir=None)
```

Primary responsibilities:

- fetch target metadata and TESS identifiers
- download and fit light curves
- extract posterior transit centers
- derive transit epochs and TTV residuals

Common call sequence in the current notebook-era workflow:

1. `get_parameter()`
2. `download_data()` / `de(...)`
3. `fit_singles()`
4. `get_posterior_samples()`
5. `calculate_ttv()`
6. `plot_tcs()` / `plot_ttv_residuals()` (`plot_ttv_re()` remains available as the legacy alias)

Operational note: this is the most environment-sensitive surface because it depends on the PyTransit stack.

### `cmat.TTVSimulation`

Legacy alias: `cmat.ttv_sim`

Constructor:

```python
TTVSimulation(epochs, ttv_mcmc, ttv_err, rs, mp2s, prop, N=80, scoring_backend=None)
```

Key constructor inputs:

- `epochs` - observed transit epochs
- `ttv_mcmc` - observed timing residuals in seconds
- `ttv_err` - timing uncertainties in seconds
- `rs` - companion-to-primary period-ratio grid
- `mp2s` - companion-mass grid
- `prop` - list containing at least one target-property dictionary
- `N` - number of transit simulations to generate

Important attributes and methods:

| Name | Kind | Purpose |
| --- | --- | --- |
| `calculate_rebound((r, mp2))` | method | Simulate one REBOUND TTV series for a single grid point |
| `get_ttv_rebound_all(number_of_thread)` | method | Run REBOUND across the full grid in parallel |
| `get_critical_masses()` | method | Preferred public alias for the current mass-threshold extraction |
| `get_m_crit()` | method | Legacy-compatible mass-threshold name |
| `get_scoring_summary()` | method | Return a JSON-serializable summary of the latest scoring result |
| `simulation_m((r, mp2))` | method | Run one MEGNO simulation |
| `run_megno(number_of_threads)` | method | Run MEGNO across the full grid |
| `plot_megno()` | method | Plot the MEGNO map |
| `scoring_backend` | attribute | Backend object used to extract critical-mass curves from the current TTV grid |
| `mass_thresholds` | attribute | Latest structured `MassThresholds` result after scoring runs |
| `megno_dt` / `megno_runtime` | attributes | Controls for MEGNO timestep and integration runtime |

`get_critical_masses()` and `get_m_crit()` return the same pair of arrays: the first rejected masses under the current `chi^2` threshold and the first rejected masses under the current RMS threshold. A period-ratio column only contributes an entry if the reduced grid actually crosses the corresponding rejection criterion, and grid points whose REBOUND integration terminated early are excluded explicitly instead of being treated as finite constraints.

## Stability note

The configuration and workflow modules are the preferred boundary for future refactors. The lower-level fitting and simulation internals should still be treated as rebuild-era APIs rather than a finalized stable interface.
