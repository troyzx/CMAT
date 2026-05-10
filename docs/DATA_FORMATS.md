# Data Formats

## Repository example data

The bundled example data live under:

```text
data/WASP-44 b/
```

Current tracked files include:

- `prop_data.csv` - cached target and system properties
- `tc_data.csv` - cached transit-timing residuals and timing uncertainties
- `mastDownload/TESS/.../*.fits` - cached TESS products used by the notebook workflow

## `tc_data.csv`

Current columns:

- unnamed row index column
- `ttv_mcmc` - inferred transit timing residual in seconds
- `ttv_err` - timing uncertainty in seconds
- `epochs` - integer transit epoch index

This file is the lightest cached input for the reduced notebook smoke path.

## `prop_data.csv`

This cached table contains target and system metadata used by the notebook and forward simulation, including:

- stellar radius and mass (`Rs`, `Ms`)
- planetary radius and mass (`Rp`, `Mp`)
- orbital period and semimajor axis (`orbital_period`, `orbital_distance`)
- transit timing reference fields

The reduced notebook smoke path currently reads:

- `orbital_distance`
- `orbital_period`
- `Mp`
- `Ms`
- `Rs`
- `Rp`

## FITS products

The cached FITS files under `mastDownload/TESS/` are the notebook's example TESS products. They remain part of the narrative workflow, but the reduced validation path intentionally avoids requiring a full transit-fitting rerun on them.
