# Data Formats

## Repository example data

The bundled example data live under:

```text
data/WASP-44 b/
```

Current tracked files include:

| Path | Purpose | Used by |
| --- | --- | --- |
| `data/WASP-44 b/prop_data.csv` | Cached target and system properties | notebook workflow, reduced forward-simulation path |
| `data/WASP-44 b/tc_data.csv` | Cached transit-timing residuals and timing uncertainties | reduced smoke test, reduced forward-simulation path |
| `data/WASP-44 b/mastDownload/TESS/.../*.fits` | Cached TESS products | full notebook-era fitting workflow |

## `tc_data.csv`

This is the lightest cached input for the reduced notebook smoke path.

Current columns:

| Column | Type | Units | Meaning | Notes |
| --- | --- | --- | --- | --- |
| *(unnamed first column)* | integer-like | none | Row index written by the original pandas export | Safe to ignore, or read as an index column |
| `ttv_mcmc` | float | seconds | Inferred transit timing residual | Centered residual series used by the current scoring path |
| `ttv_err` | float | seconds | Timing uncertainty for each observed transit | Used directly by the current `chi^2` scoring path |
| `epochs` | integer | transit count | Transit epoch index | Must stay aligned with `ttv_mcmc` and `ttv_err` |

Typical workflow use:

- `epochs` indexes the observed transit events
- `ttv_mcmc` provides the residual series used for TTV comparison
- `ttv_err` provides the timing uncertainty used by the current `chi^2` scoring path

The reduced smoke path and synthetic examples assume all three scientific columns can be loaded into one-dimensional NumPy arrays of equal length.

## `prop_data.csv`

This cached table contains target and system metadata used by the notebook and forward simulation. The repository example is a wide, metadata-rich table rather than a minimal simulation-only schema.

The current file follows a repeated field-family pattern:

| Field family | Meaning |
| --- | --- |
| `<field>` | Nominal numeric value |
| `<field>_unit` | Unit label stored in the CSV |
| `<field>_upper` / `<field>_lower` | Upper and lower uncertainty values |
| `<field>_ref` | Reference citation string |
| `<field>_url` | Source URL |

Examples of these families in the current repository data include stellar parameters, planetary parameters, orbital parameters, photometric magnitudes, and transit-reference metadata.

### Minimal fields required by the current forward-simulation path

The reduced notebook smoke path currently reads only the following numeric fields:

| Field | Example unit in repository data | Meaning | Required by |
| --- | --- | --- | --- |
| `orbital_distance` | `AU` | Semimajor axis of the observed planet | `ttv_sim` setup |
| `orbital_period` | `d` | Orbital period of the observed planet | forward simulation context |
| `Mp` | `M_Jupiter` | Mass of the observed planet | REBOUND primary-planet setup |
| `Ms` | `M_sun` | Stellar mass | REBOUND stellar mass |
| `Rs` | `R_sun` | Stellar radius | transit geometry / collision radius |
| `Rp` | `R_Jupiter` | Radius of the observed planet | transit geometry |

These six values are enough to instantiate the current `ttv_sim` forward model on cached data.

### Minimal in-memory `prop` shape

The current simulation API expects `prop` as a list whose first entry is a dictionary containing at least the fields above:

```python
prop = [
    {
        "orbital_distance": 0.0347371,
        "orbital_period": 2.4238039,
        "Mp": 0.889836,
        "Ms": 0.951,
        "Rs": 0.927,
        "Rp": 1.14,
    }
]
```

The full notebook-era workflow stores many more keys, but the reduced forward-simulation path only requires this subset.

### Additional target metadata in the repository example

The shipped `prop_data.csv` also includes many fields that are currently documented only informally, including:

- target naming and catalog identifiers (`canonical_name`, `planet_name`, `star_name`, `catalog_name`)
- stellar atmosphere and magnitude fields (`Teff`, `Vmag`, `Jmag`, `Hmag`, `Kmag`, `Tmag`)
- orbital geometry (`inclination`, `eccentricity`, `omega`, `a/Rs`, `impact_parameter`)
- transit observables (`transit_time`, `transit_duration`, `transit_depth`, `Rp/Rs`)
- provenance fields (`*_ref`, `*_url`)

That richer surface is still useful for the full notebook narrative, but it is intentionally outside the smallest validated quick-start contract.

## FITS products

The cached FITS files under `mastDownload/TESS/` are the notebook's example TESS products. They remain part of the narrative workflow, but the reduced validation path intentionally avoids requiring a full transit-fitting rerun on them.

At the moment, the rebuild documents these FITS inputs at the directory level rather than as a frozen schema. That is deliberate: Stage 3 aims to make the light-weight path clear first, while the full notebook-era fitting inputs are still being carved into a cleaner library boundary.

## Workflow manifest shape

The new configuration layer also introduces a JSON-serializable manifest shape via `cmat.workflow.workflow_manifest(...)`. The current object is:

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `config` | object | yes | Serialized `RunConfig` |
| `code_version` | object | no | Git commit and dirty-state metadata when available |
| `dependency_versions` | object | no | Version map for pinned runtime dependencies |
| `runtime` | object | no | Timestamp, Python/platform, cwd, and selected runtime environment |
| `notes` | object | no | Free-form run notes |
| `scoring_summary` | object | no | Serialized scoring result surface |

The associated output-path helper lives in `OutputConfig.metadata_path`, which defaults to:

```text
artifacts/run_metadata.json
```

CMAT can now persist this manifest explicitly via `cmat.workflow.write_workflow_manifest(...)`. The notebook-driven workflow still does not auto-write it on every path, but the serialized shape and default output location are now part of the library boundary.

## Cache artifact shapes

Stage 4 also adds explicit cache helpers for expensive intermediate products.

| Path | Format | Contents |
| --- | --- | --- |
| `artifacts/cache/ttv_grid.npz` | NumPy compressed archive | `period_ratios`, `companion_masses`, `epochs`, observed `ttv_mcmc` / `ttv_err`, and cached `ttv_results` |
| `artifacts/cache/megno_grid.npz` | NumPy compressed archive | `period_ratios`, `companion_masses`, and cached `megno_results` |
| `artifacts/cache/posterior_samples.json` | JSON | Retained Bayesian posterior samples plus compact sampler/reference metadata |

When `run_name` is set, these files live under `artifacts/<run_name>/cache/`.

## Stability note

The current data-format coverage is intentionally pragmatic rather than exhaustive. The reduced CSV-driven path is now explicit, but the rebuild still needs a more formal schema description for the notebook-era FITS inputs and any future persisted result tables.
