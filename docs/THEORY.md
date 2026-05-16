# Theory

CMAT constrains hidden companion masses from transit timing variation (TTV) observations.

## Workflow model

1. Fit a transiting planet's light curve.
2. Estimate per-transit center times and uncertainties.
3. Convert those estimates into epoch-indexed TTV residuals.
4. Simulate candidate two-planet systems with REBOUND.
5. Compare simulated TTVs with observed residuals.
6. Use MEGNO as an additional stability diagnostic.

## Inverse-modeling framing

The same workflow can be described without astronomy-specific jargon:

| Workflow layer | CMAT currently does | Transferable role |
| --- | --- | --- |
| Observation extraction | Reduces light curves into per-transit timing measurements | Converts raw measurements into a compact observation vector |
| Uncertainty estimation | Carries timing posteriors and `ttv_err` into downstream scoring | Makes data ambiguity explicit before simulation and rejection |
| Latent-state parameterization | Uses companion mass and period ratio as the hidden-state grid | Defines the structured hypothesis space |
| Forward simulation | Uses REBOUND to map each latent state to synthetic TTV observables | Predicts observation-space consequences of each hypothesis |
| Hypothesis rejection | Scores mismatch with chi2/RMS or the experimental Bayesian summary backend | Removes inconsistent latent states |
| Decision-surface generation | Produces mass-limit curves plus MEGNO stability context | Summarizes the viable vs rejected region of parameter space |

This is the main industry-facing interpretation of CMAT: not “an astronomy notebook that happens to simulate planets”, but a compact example of uncertainty-aware inverse modeling with a physics-based forward simulator and an interpretable rejection surface.

## Observation model

The observed quantity is not the unseen companion directly. Instead, CMAT uses sparse timing residuals inferred from repeated transit events.

## Forward model

The current forward model is a grid of REBOUND N-body simulations over:

- companion period ratios `P2/P1`
- companion masses `M2`

## Current scoring

The current baseline scoring compares simulated and inferred TTVs using:

- `chi^2` over epoch-aligned simulated residual windows
- RMS amplitude screening

Stage 2 validation now includes inject-recovery tests for these scoring paths. A more explicit probabilistic scoring model is deferred to Stage 4 of the rebuild.

Stage 4 now also includes an experimental Bayesian posterior mass-summary backend. Its primary role is to add explicit nuisance-parameter marginalization and evidence-backed mass summaries without changing the legacy chi2/RMS baseline contract.

## Stability interpretation

MEGNO is used as a dynamical-stability diagnostic alongside the TTV rejection surface. A low-mass candidate that matches the timing data may still be dynamically implausible if the corresponding MEGNO region is strongly unstable.

## Interpreting outputs

### Mass-limit curves

The critical-mass output is best read as a rejection boundary, not a full posterior over all unseen companions. A lower critical mass at a given period ratio means the observed timing data rule out smaller companions there.

### MEGNO maps

MEGNO is a stability diagnostic, not a direct fit-quality score. In practice:

- a region can be rejected by TTV mismatch even if it is dynamically stable
- a region can look compatible in TTV space but still be dynamically unstable

The most interesting regions are usually the ones that remain both observationally viable and dynamically plausible.

### Observational caveats

Current rejection boundaries still depend on:

- the chosen simulation grid
- the current `chi^2` and RMS scoring rules
- the finite set of measured transit times and their uncertainties

That is why the Stage 2 rebuild adds inject-recovery checks before any scoring backend is replaced.
