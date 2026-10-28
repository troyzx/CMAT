# CCS — Cold Companion Search (research foundation, 2026-10-28)

**Goal:** Develop a reproducible, sample-level search for nontransiting outer
companions to well-timed transiting planets using the **dynamical TTV + stellar
light-travel-time effect (LTTE)**. Expand CMAT beyond mass-upper-limit maps to
model comparison and, ultimately, validated orbital-parameter inference.

**Status: FOUNDATION/PILOT ONLY. NO DISCOVERY OR CONFIDENCE CLAIMS.**

This code belongs to `cmat.ccs` to ensure it is installed by CMAT's existing
`[tool.setuptools.packages.find] include = ["cmat*"]`. It does not modify
existing `cmat.ttv_sim`, old paper scores, CMAT `main`, or the `astro-data` DVC
repository. Python 3.10/3.11 is the supported CMAT rebuild environment.

## Repository roles and audited baselines

| Source | Role | Audited commit |
| --- | --- | --- |
| `troyzx/CMAT` `dev/industry-rebuild` | reusable package, score/cache/REBOUND modules | `5e190aa5d0c74f53c06fdeeba843d00d1c3b9145` |
| `troyzx/CMAT` `main` | contains merged MEGNO/unit-label PR #18 | `9bfbd007f832d3bbc74d6145c3d8728d45cb221d` |
| `troyzx/astro-data` `main` | DVC data and secondary-eclipse pipeline | `fae85f61a6c212ada681f0658aba6f770d8d3eed` |
| `troyzx/wd1856-ttv-romer` | LTTE/linear-ephemeris physics reference | independently maintained |

See `DATA_AUDIT.md` for limitations: GitHub `.dvc` pointers do not prove local
raw timing assets are in cache. The newer eclipse project's 455 usable systems
are **not** the old TTV paper's 260 measured timing datasets.

## What works in this first commit

- Strict canonical transit-center ingestion: `epoch,t_mid_bjd_tdb,sigma_sec`;
  duplicates, reference, instrument and timing-group labels are preserved.
- Weighted reference linear ephemeris for numerical conditioning; each model
  scores with `fixed`, `offset`, `median` (legacy diagnostic), or `linear`
  (method C: two profiled nuisance parameters).
- Circular LTTE formula for independent, cheap sensitivity estimates.
- Limited REBOUND pilot: two massive planets, circular/copolanar/edge-on,
  raw transit event timestamps, stellar barycentric z, four signals:
  M0 zero, M1 dynamical, M2 LTTE, M3 joint.
- `pilot` only evaluates ONE outer mass-period-phase point. It does **not**
  search or optimize outer parameters, calibrate significance, compute complete
  mass limits, or infer companion occurrence. No bulk runner is enabled.

### Scientific warning about the ephemeris

The C method fits an additive `a + b(E-E_ref)` after each integration. This
is **not** the same as re-integrating with a different *osculating* period.
For a publishable candidate, include `P_inner,osc` and physical initial epoch
in a joint numerical optimization, and verify the result after reintegration.
The reference ephemeris computed from the observations is a coordinate choice,
not an independent prior to be used twice.

The idealized analytic circular LTTE is

`A_LTT = (a_outer/c) * (m_outer/M_total) * sin(i_outer)`.

The pilot uses `-z_star/c` at numerically determined transit events, while
subtracting a no-outer baseline with the same massive inner planet. The
numerical model includes the inner planet's mass. Stellar z and transit event
finding **still need REBOUND runtime validation** on the user's machine.

## Quick start (read-only, no DVC pull)

From the CMAT repository root on `research/ccs-foundation-20261028`:

```bash
# Use the existing constrained Python 3.10/3.11 CMAT environment if available.
python -m unittest discover -v -s tests -p 'test_ccs_*.py'
python -m cmat.ccs demo
```

The `demo` uses deterministic synthetic data and needs NumPy only. Full CMAT
installation constraints are in `pyproject.toml` and `constraints.txt`.

For a *verified* canonical timing CSV:

```bash
python -m cmat.ccs audit --timings /path/to/verified_timings.csv
```

To explicitly convert a legacy CSV known to be BJD_TDB with uncertainty in
**days**, run (will never overwrite an existing output):

```bash
python -m cmat.ccs convert-legacy \
  --source /path/to/legacy.csv \
  --out /path/to/converted_timings.csv --assume-bjd-tdb
```

This conversion DOES NOT establish that a legacy timestamp was really BJD_TDB.
Verify provenance, exposure-midpoint corrections, and time scale first.

For a small real/pilot timing dataset with independently verified stellar and
inner-planet masses (REBOUND required):

```bash
python -m cmat.ccs pilot \
  --timings /path/to/verified_timings.csv \
  --star-mass-msun 1.0 --inner-mass-mjup 1.0 \
  --outer-mass-mjup 1.0 --outer-period-day 100.0 \
  --outer-phase-deg 90.0 --mode linear \
  --out /path/to/new_pilot_result.json
```

Above numbers are placeholders. Do **not** use them as actual target
parameters. Pilot maximum epoch span is 3000 by default; it refuses larger
spans and overlong event stepping. Existing outputs are never overwritten.

## Intended output and acceptance

Every single-model pilot emits the canonical data SHA, reference ephemeris,
input model parameters and same-noise M0/M1/M2/M3 scores. `delta_chi2` here is
a *conditional fit diagnostic*, not a detection significance or Bayes factor.

Before interpreting any real target: pass numerical REBOUND tests, compare
REBOUND star reflex against analytic LTTE, test timestep convergence, investigate
data covariance and initial-period optimization, and calibrate the full search
with null simulations and injected companions.

See `CODEX_HANDOFF.md` for the first local Codex acceptance sequence.
