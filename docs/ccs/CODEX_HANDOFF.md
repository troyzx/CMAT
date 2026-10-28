# Local Codex — CCS WP0 + physics smoke acceptance

**Repository:** `troyzx/CMAT`.
**Target branch:** `research/ccs-foundation-20261028`.
**Parent:** `dev/industry-rebuild` at
`5e190aa5d0c74f53c06fdeeba843d00d1c3b9145`.

## Instructions to Codex (copy/paste)

Work on the `research/ccs-foundation-20261028` branch in CMAT. First inspect
`git status`, source branch, and `docs/ccs/{README,DATA_AUDIT,PROJECT_PLAN}.md`.
Do not modify CMAT main, existing science results, `astro-data` or DVC remote.
No DVC network pull/push and no large N-body grid. Use only test/synthetic data
until validation is complete.

1. Install the CMAT constrained Python 3.10/3.11 environment if missing;
   respect `constraints.txt` and existing local venv rather than changing
   unrelated pinned dependencies. Inspect `pyproject.toml`.
2. Run `python -m unittest discover -v -s tests -p 'test_ccs_*.py'` and
   `python -m cmat.ccs demo`. Record exact successes, failures and skips.
3. In an environment with `rebound==3.26.3`, require
   `tests.test_ccs_dynamics` to PASS; if REBOUND is missing it is a blocker,
   not success. Compare pilot numeric star-z LTTE against analytic orbit and
   test the transit crossing and epoch labels across positive/negative epochs.
4. Perform a numerical integration-step convergence check for M1/M2/M3,
   focusing on ~0.1 s and better predictions; do NOT claim millisecond
   precision unless demonstrated. Verify time unit, BJD_TDB, AU/day c, mass
   unit conversion, massive inner planet, coordinate sign and Jacobi convention.
5. Review the prior CMAT refactor scoring: old epoch-shift χ² is NOT C
   method; old REBOUND returns detrended TTV and cannot be used to generate
   LTTE by adding it back naively. Keep legacy modules unchanged.
6. Audit the *local* `astro-data` DVC metadata/cache read-only; locate original
   260-target TTV transit centers, uncertainty/time-scale provenance, and
   compare to newer eclipse target list. Produce `docs/ccs/LOCAL_DATA_AUDIT.md`
   with exact verified paths/hashes and explicit unknowns. No implicit
   adoption of 455 eclipse targets as 455 TTV series.
7. Prepare a small synthetic injection/recovery and a no-companion null test
   for the next PR, but do not launch a mass-period search. Decide if full
   initial-period reintegration or photodynamic timing corrections are needed.

Finish with a concise Chinese report: pass/fail of each test, physics blockers,
verified data assets, numerical runtime, next smallest WP2 implementation step.
Push only changes in this research branch after tests; no merge without review.

## Commands

```bash
git fetch origin
git switch research/ccs-foundation-20261028
python -m unittest discover -v -s tests -p 'test_ccs_*.py'
python -m cmat.ccs demo
python -m cmat.ccs --help
```

Do not run `pilot` on 81 real WD1856 or all 260 systems until standalone
REBOUND transit-time and stellar-z unit tests have passed on the compute host.
