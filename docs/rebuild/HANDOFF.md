# CMAT Industry Rebuild Handoff

Generated: 2026-05-10

This document is a self-contained handoff for continuing the CMAT to Industry
rebuild. It records the current branch strategy, completed work, validation
state, known risks, and next recommended steps.

## Task

Continue the staged rebuild of CMAT from an astronomy notebook-driven package
into a more maintainable scientific Python project that can also be presented
as Bayesian inverse modeling, simulation-based inference, uncertainty
quantification, sparse time-series analysis, and physics-based forward
simulation.

Preserve the current scientific behavior unless regression tests explicitly
protect the behavior being refactored.

## Branch and PR Model

- Keep `main` stable.
- Use `dev/industry-rebuild` as the long-lived integration branch for the full rebuild.
- Open each rebuild stage as a PR into `dev/industry-rebuild`, not into `main`.
- After all stages are reviewed and integrated into `dev/industry-rebuild`, open one final PR from `dev/industry-rebuild` into `main`.

Current branches:

- `main`: stable branch, currently at `9015436`.
- `dev/industry-rebuild`: integration branch, created from `main` at `9015436`.
- `codex/industry-rebuild-stage-1-packaging`: active Stage 1 branch.

Current PR:

- PR #5: `codex/industry-rebuild-stage-1-packaging` -> `dev/industry-rebuild`
- URL: <https://github.com/troyzx/CMAT/pull/5>
- Status at handoff: open draft, mergeable, and still targeting `dev/industry-rebuild`.
- The branch now extends the original Stage 1 packaging/configuration foundation with early Stage 2 validation coverage and initial CI scaffolding.

## Completed Work

Already merged before this handoff:

- README rewrite for the CMAT to Industry presentation.
- Stage 0 baseline and scope control.
- Stage 0 environment and coverage expansion.

Open in PR #5:

- Added `pyproject.toml` with PEP 621 package metadata.
- Added tracked `.gitignore`.
- Updated install guidance to use editable package installation with constraints.
- Added lazy top-level package exports so `import cmat` avoids eager PyTransit import.
- Added typed configuration objects in `cmat.config`.
- Added thin workflow adapters in `cmat.workflow`.
- Extracted TTV scoring helpers into `cmat.scoring` while preserving legacy imports from `cmat.ttv_sim`.
- Added tests for package import behavior, configuration objects, workflow adapters, scoring behavior, and legacy scoring imports.
- Added Stage 1 rebuild documentation.
- Added regression coverage for epoch derivation and mass-threshold extraction edge cases.
- Added a reduced deterministic REBOUND regression test.
- Fixed `ttv_sim.get_m_crit` so zero-signal simulation rows are skipped instead of being counted as rejected masses.
- Added initial GitHub Actions CI scaffolding for constrained editable installs, `pip check`, `compileall`, and `unittest` on Python 3.10 and 3.11.
- Recorded the reduced notebook smoke-test plan in the existing rebuild docs.

Stage 1 commits on the active branch:

- `7b2c09e build: add pyproject packaging metadata`
- `ceb2870 refactor: add typed workflow configuration`
- `f51759f refactor: add workflow configuration adapters`
- `5f0cf51 refactor: extract ttv scoring helpers`

Latest local additions after the original Stage 1 commit stack:

- focused Stage 2 validation and REBOUND regression coverage
- initial CI workflow scaffolding
- updated rebuild roadmap, limitations, and handoff notes

## Relevant Files

- `README.md` - public project page rewritten for the industry-readable framing.
- `TODO_industry_rebuild.md` - staged roadmap.
- `constraints.txt` - temporary dependency guardrail for the rebuild baseline.
- `pyproject.toml` - tracked packaging metadata on Stage 1 branch.
- `cmat/__init__.py` - lazy top-level exports.
- `cmat/config.py` - typed configuration objects.
- `cmat/workflow.py` - adapters from configuration objects to existing implementation.
- `cmat/scoring.py` - extracted TTV scoring helpers.
- `cmat/ttv_sim.py` - REBOUND forward simulation; still preserves legacy scoring imports.
- `.github/workflows/ci.yml` - constrained editable-install CI on Python 3.10 and 3.11.
- `docs/rebuild/CURRENT_LIMITATIONS.md` - known limitations and unresolved debt.
- `docs/rebuild/ENVIRONMENT_BASELINE.md` - environment findings and install guidance.
- `docs/rebuild/STAGE1_PACKAGING.md` - packaging validation notes.
- `docs/rebuild/STAGE1_PROJECT_STRUCTURE.md` - configuration, workflow, and scoring split notes.
- `docs/rebuild/HANDOFF.md` - updated branch, validation, and next-stage continuation notes.
- `tests/` - baseline and Stage 1 regression tests.
- `tests/test_ttv_rebound.py` - reduced deterministic REBOUND regression fixture.

## Validation Snapshot

Validated on the Stage 1 branch after the latest local changes:

```bash
git diff --check
python -m compileall cmat
MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
XDG_CACHE_HOME=/private/tmp/cmat-cache \
/private/tmp/cmat-rebuild-env/bin/python -m pip check
MPLCONFIGDIR=/private/tmp/cmat-mplconfig \
XDG_CACHE_HOME=/private/tmp/cmat-cache \
/private/tmp/cmat-rebuild-env/bin/python -m unittest discover -v -s tests
```

Results:

- `git diff --check`: passed.
- `python -m compileall cmat`: passed.
- Constrained disposable environment: `pip check` passed and 25 tests passed.
- Existing global environment still has the known PyTransit/Numba/llvmlite mismatch; treat the constrained environment as authoritative unless that path also regresses.

Earlier Stage 1 packaging validation also passed:

```bash
python -m pip check
python -m pip wheel --no-deps . -w /private/tmp/cmat-wheelhouse -c constraints.txt
```

## Environment Notes

The constrained environment is the reliable validation path:

```bash
python -m venv /private/tmp/cmat-rebuild-env
/private/tmp/cmat-rebuild-env/bin/python -m pip install -e . -c constraints.txt
```

Use writable cache locations when running tests in sandboxed contexts:

```bash
MPLCONFIGDIR=/private/tmp/cmat-mplconfig
XDG_CACHE_HOME=/private/tmp/cmat-cache
```

Known environment issue:

- The existing global Python 3.11 environment has an incompatible PyTransit/Numba/llvmlite import path.
- The constrained environment fixes the full import stack.
- Do not treat the global skip as a new Stage 1 regression unless the constrained environment also fails.

## Known Local State

There is one old local stash:

```text
stash@{0}: On cmat-industry-rebuild: wip pre-rebuild singlefit and notebook metadata
```

Do not apply it casually. It contains pre-rebuild WIP around `cmat/singlefit.py`
and notebook metadata. Earlier inspection found unfinished code in that stash,
including undefined names. Keep it separate unless the user explicitly asks to
recover or triage it.

Ignored local artifacts may exist in the workspace:

- `.DS_Store`
- `.ipynb_checkpoints/`
- `build/`
- `dist/`
- `CMAT_astro.egg-info/`
- local legacy `setup.py`

These are not part of the tracked rebuild baseline.

## Decisions

- Use a long-lived `dev/industry-rebuild` branch for staged integration.
- Keep Stage PRs reviewable and merge them into `dev/industry-rebuild`, not `main`.
- Keep PR #5 as a draft while Stage 1 is still being shaped.
- Add configuration and workflow boundaries before deeper algorithm-facing refactors.
- Preserve legacy public imports where practical during refactoring.
- Do not change scientific formulas, fitting behavior, TTV residual construction, REBOUND integration behavior, or mass-threshold logic without regression coverage.
- Prefer standard-library `unittest` for current tests to avoid adding test dependencies prematurely.

## What Was Tried

- The initial PR creation through local `gh` was blocked by invalid CLI auth, but the GitHub connector succeeded.
- Creating `dev/industry-rebuild` required elevated Git permissions because local `.git` ref writes were sandbox-restricted.
- A package import test initially depended on `cmat.ttv_sim` staying a callable package attribute after the submodule was imported. That assumption was incorrect because Python normally sets `cmat.ttv_sim` to the imported submodule. The test was corrected to verify the lazy accessor directly.

## Constraints

- Scope remains controlled: documentation, packaging, tests, API boundaries, and safe refactors first.
- No algorithmic changes without tests that pin the current scientific behavior.
- Preserve the astronomy use case: upper mass constraints for hidden companions from TTV observations.
- Preserve broader framing: inverse modeling, simulation-based inference, uncertainty quantification, sparse time-series analysis, and physics-based forward simulation.
- Keep communication and docs serious, concise, and technically credible.

## Recommended Next Steps

1. Finish reviewing PR #5 as the Stage 1 packaging and structure foundation.
2. Decide whether Stage 1 should add a small seed/logging utility before marking the PR ready.
3. Merge PR #5 into `dev/industry-rebuild` only after review.
4. Create the next branch from `dev/industry-rebuild`, likely `codex/industry-rebuild-stage-2-validation`.
5. Continue Stage 2 from the updated baseline:
   - add small synthetic-system tests with injected timing offsets and known recovery expectations;
   - convert the notebook smoke-test plan into an automated reduced execution path;
   - add MEGNO regression coverage or a reduced deterministic guardrail;
   - expand CI from constrained unit tests into notebook smoke execution and linting if the dependency story stays stable.
6. After the remaining Stage 2 work is in place, move into documentation/examples and only then larger inference/performance refactors.

## Acceptance Criteria For The Next Agent

- The current branch and PR state are understood before making edits.
- New work targets `dev/industry-rebuild`, not `main`.
- The constrained environment remains the authoritative validation environment.
- Existing tests pass before pushing.
- The constrained CI workflow is treated as the automated baseline, while notebook smoke execution remains planned but not yet automated.
- Any source refactor preserves legacy behavior or adds regression tests that justify the change.
- The old stash remains untouched unless explicitly requested.

## Handoff Prompt

Use this prompt for a fresh agent:

```text
## Task
Continue the CMAT to Industry rebuild from the current repository state. Work in staged PRs targeting `dev/industry-rebuild`, not `main`.

## Context
CMAT is an astronomy Python project for constraining hidden companion masses from TTV observations. The rebuild should keep the science accurate while presenting and structuring the project as Bayesian inverse modeling, simulation-based inference, uncertainty quantification, sparse time-series analysis, and physics-based forward simulation.

README rewrite and Stage 0 baseline work are already merged to `main`. A long-lived integration branch, `dev/industry-rebuild`, exists. Stage 1 is open as draft PR #5 from `codex/industry-rebuild-stage-1-packaging` into `dev/industry-rebuild`.

## Relevant Files
- `TODO_industry_rebuild.md` - staged roadmap.
- `docs/rebuild/HANDOFF.md` - current handoff.
- `docs/rebuild/CURRENT_LIMITATIONS.md` - known limitations.
- `docs/rebuild/ENVIRONMENT_BASELINE.md` - dependency and environment baseline.
- `docs/rebuild/STAGE1_PACKAGING.md` - Stage 1 packaging notes.
- `docs/rebuild/STAGE1_PROJECT_STRUCTURE.md` - Stage 1 structure notes.
- `pyproject.toml` - new package metadata on the Stage 1 branch.
- `constraints.txt` - constrained validation dependency guardrail.
- `cmat/config.py` - typed configuration objects.
- `cmat/workflow.py` - thin workflow adapters.
- `cmat/scoring.py` - scoring helpers extracted from `ttv_sim`.
- `tests/` - current regression tests.

## Current State
PR #5 currently includes packaging metadata, lazy imports, configuration objects, workflow adapters, scoring helper extraction, and associated tests. The constrained venv passes 21 tests. The global environment passes 20 tests and skips one known PyTransit/Numba/llvmlite import test.

## What Was Tried
- Local `gh` auth failed earlier, but the GitHub connector created and updated PR #5.
- Creating `dev/industry-rebuild` and commits required elevated Git permissions because sandboxed `.git` writes were restricted.
- A package import test was corrected after recognizing that importing `cmat.ttv_sim` as a submodule normally changes the package attribute.

## Decisions
- Use `dev/industry-rebuild` as the integration branch.
- Merge stage PRs into `dev/industry-rebuild`, not `main`.
- Preserve legacy public imports while refactoring.
- Do not change scientific behavior without regression coverage.
- Use the constrained disposable environment as the authoritative validation path.

## Acceptance Criteria
- [ ] Work starts from the correct branch/base.
- [ ] Existing tests pass in the constrained environment.
- [ ] Any new refactor is covered by focused tests.
- [ ] No old stash is applied unless explicitly requested.
- [ ] PR descriptions include validation results and target `dev/industry-rebuild`.

## Constraints
Do not modify algorithms unless the user explicitly scopes that work and regression tests protect current behavior. Keep the documentation and commit messages concise, serious, and technically credible.
```
