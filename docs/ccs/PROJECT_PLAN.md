# CCS research plan / scientific decision gates

**Question:** Can a common transit-timing forward model, combining N-body
perturbations and stellar light-travel-time delay, find or constrain colder,
nontransiting companions around well-timed transiting planets?

## Models and estimand

- M0: linear ephemeris, no external companion.
- M1: gravitational dynamical transit-time shift from an outer companion.
- M2: stellar barycentric LTTE alone.
- M3: dynamical transit-time shift + LTTE, fully consistent event epochs.
- Profile an additive timing zero-point and mean period for every model;
  additionally retain fixed/median scores as controlled method comparisons.
- Don't misinterpret one grid-cell minimum as a detected planet. Scanning mass,
  period and phase has a look-elsewhere effect, and the companion-free limit
  has unidentified orbital nuisance parameters. Simulate the full null search.

## Ordered work packages

| WP | Scope | Gate |
|---|---|---|
| WP0 | Inventory 260 TTV assets, DVC pointers, CMAT refactor, WD1856 | Frozen provenance ledger; no missing data silently inferred |
| WP1 | Canonical mid-time data with epochs, BJD_TDB, seconds, groups | Duplicate and uncertainty validation; 5-10 pilot targets selected by timing suitability |
| WP2 | Physics: raw REBOUND times, stellar reflex LTTE, positive inner mass | Circular analytic, independent barycentric reference, epoch, step convergence checks |
| WP3 | Likelihood/optimizer: M0-M3, linear profile, full-P reintegration | Injected mass/phase recovery, alternate starting points, nuisance checks |
| WP4 | 5-10 targets, coarse/fine region search | Reproducible scores, frozen inputs, no uncalibrated discovery claims |
| WP5 | Complete time-domain null bootstrap, injection/recovery, stability and RV/Gaia cross-check | Validated false-alarm rate and search completeness |
| WP6 | Expand eligible sample, population constraints and paper | Selection function, uncertainty and science limitations explicit |

## First integration targets

- Recoverable historical 260 target mid-time observations, not re-fit all TESS
  photometry by default.
- WD1856 serves as an independent LTTE method benchmark but cannot validate
  a massive-inner-planet model by itself.
- Explicit *zero-outer-companion* tests: M0 is the reference, not an external
  prior from the same 81 timings.
- Instrument systematics and jitter are later WP3/WP4 experiments. Jitter
  likelihood must include covariance log determinant.

## Scientific caveats and required extensions

The pilot assumes 2 planets, circular/coplanar/edge-on, an exact stellar mass,
no instrument offsets, Gaussian independent given sigmas, and a fixed initial
inner osculating P. Outer inclination, eccentricity, light-time convention,
long-term dynamical stability, finite transit geometry and full photodynamical
model are future-work gates. Inferred mass without outer inclination may have
an `M sin i`-like degeneracy; do not claim unique 3D mass retrieval.

## Compute safety

No batch search enabled in first CCS commit. Maximum epoch span and
integration-step limits protect shared machines. New batch code must add
checkpoints, lock lifecycle, input/config/hash provenance, worker caps and a
hard wall-time guard before any full production runs. New DVC products require
separate explicit approval; no automatic `dvc pull` or `dvc push`.
