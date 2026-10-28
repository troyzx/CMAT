# CCS source/data audit — observed in remote GitHub, 2026-10-28

## Verified repository facts

- `troyzx/astro-data` is private. Its main branch is
  `fae85f61a6c212ada681f0658aba6f770d8d3eed`.
- DVC is configured with a remote named `storage` over SSH. Do NOT rewrite the
  remote. A `.dvc` pointer alone is not confirmation that its payload is in
  local cache or accessible from the workstation.
- The 2026 September secondary-eclipse sample freeze describes **455** fitted
  hot Jupiters among **458** physical targets. That sample is NOT interchangeable
  with the published earlier **260-system TTV** timing dataset.
- `TESS/cmat/`, `TESS/fitransit_local/` are legacy paths. `astro-data`'s
  current root-native pipeline is `src/tess_hj_ses/`. Avoid modifying either.
- `troyzx/CMAT`'s modularized refactor is on `dev/industry-rebuild`.
  An independent main-branch PR #18 merged on 2026-10-19 fixes MEGNO orbital
  distance and Jupiter/Earth mass labels. Reconcile before using MEGNO.

## Blocked/unknown until local audit

- Exact path and schema of all **260 original TTV mid-times**, including
  epoch, BJD time scale, errors, instruments, publication provenance.
- Availability/completeness of DVC objects in the local cache; remote auth.
- Validity and covariance of heterogeneous timing uncertainties.
- Crossmatch of original 260 TTV targets to secondary-eclipse sample v2.
- Version consistency between historical CMAT results and raw timestamps.
- Runtime REBOUND 3.26.3 validation of the new event finder and stellar LTTE.

## Local Codex WP0/WP1 acceptance

1. Inspect status and DVC configuration **read-only**, never automatically pull
   terabytes or modify cache/remote. Record `git rev-parse HEAD`, `dvc status`
   only if it is non-mutating in the local setup, and candidate `.dvc` pointers.
2. Find old 260-target timing files by actual inspection. Never assume a
   plotted O-C is equivalent to full raw BJD transit centers.
3. Produce a target ledger and raw-to-canonical field mapping, with exact
   path, SHA256, original time scale, and uncertainty units.
4. Ensure every missing field or unavailable object is marked `UNKNOWN` or
   `MISSING`, not guessed. Ask before large downloads or migration.
5. Work on CCS branch; do not change any frozen result, `astro-data` pipeline,
   other papers, DVC remote, or current scientific results.
