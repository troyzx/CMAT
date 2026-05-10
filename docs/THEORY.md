# Theory

CMAT constrains hidden companion masses from transit timing variation (TTV) observations.

## Workflow model

1. Fit a transiting planet's light curve.
2. Estimate per-transit center times and uncertainties.
3. Convert those estimates into epoch-indexed TTV residuals.
4. Simulate candidate two-planet systems with REBOUND.
5. Compare simulated TTVs with observed residuals.
6. Use MEGNO as an additional stability diagnostic.

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

## Stability interpretation

MEGNO is used as a dynamical-stability diagnostic alongside the TTV rejection surface. A low-mass candidate that matches the timing data may still be dynamically implausible if the corresponding MEGNO region is strongly unstable.
