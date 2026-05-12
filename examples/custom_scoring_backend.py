"""Demonstrate injecting a custom scoring backend into CMAT.

This example is intentionally tiny and deterministic. It does not run REBOUND;
instead, it shows how a Stage 4 scoring backend can plug into `TTVSimulation`
and produce a JSON-serializable scoring summary.
"""

from __future__ import annotations

import numpy as np

from cmat import TTVSimulation
from cmat.scoring import MassThresholds


class FixedDemoScorer:
    """Minimal scorer that returns fixed mass-threshold curves."""

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
        del ttv_results, epoch, ttv_mcmc, ttv_err, companion_masses
        return MassThresholds(
            chi2=np.full(len(period_ratios), 12.0),
            rms=np.full(len(period_ratios), 18.0),
            backend="fixed-demo",
        )


def main() -> None:
    simulation = TTVSimulation(
        epochs=np.array([0, 1, 2]),
        ttv_mcmc=np.array([0.0, 1.0, 0.0]),
        ttv_err=np.ones(3),
        rs=np.array([1.5, 2.0]),
        mp2s=np.array([10.0, 20.0]),
        prop=[
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
                "Rs": 1.0,
                "Rp": 1.0,
            }
        ],
        scoring_backend=FixedDemoScorer(),
    )
    simulation.ttv_results = [np.zeros(4)] * 4

    chi2_limit, rms_limit = simulation.get_critical_masses()

    print("Custom scoring backend example")
    print("chi2 limits:", chi2_limit.tolist())
    print("rms limits:", rms_limit.tolist())
    print("scoring summary:", simulation.get_scoring_summary())


if __name__ == "__main__":
    main()
