"""Run a small synthetic CMAT forward-simulation example.

This example avoids remote data access and transit fitting. It starts from a
small synthetic TTV series and exercises the reduced forward-simulation path
used in the Stage 3 quick-start documentation.
"""

from __future__ import annotations

import numpy as np

from cmat.ttv_sim import ttv_sim


def main() -> None:
    epochs = np.array([0, 1, 2, 3])
    ttv_mcmc = np.array([12.0, -8.0, 15.0, -10.0])
    ttv_err = np.full(4, 5.0)
    prop = [
        {
            "orbital_distance": 0.055,
            "orbital_period": 4.0,
            "Mp": 1.0,
            "Ms": 1.0,
            "Rs": 1.0,
            "Rp": 1.0,
        }
    ]

    simulation = ttv_sim(
        epochs=epochs,
        ttv_mcmc=ttv_mcmc,
        ttv_err=ttv_err,
        rs=np.array([1.5, 2.0]),
        mp2s=np.array([10.0, 20.0]),
        prop=prop,
    )
    simulation.ttv_results = [
        simulation.calculate_rebound((period_ratio, companion_mass))
        for companion_mass in simulation.mp2s
        for period_ratio in simulation.rs
    ]
    chi2_limit, rms_limit = simulation.get_m_crit()

    simulation.megno_dt = 0.02
    simulation.megno_runtime = 50.0
    megno = simulation.simulation_m((1.5, 10.0))

    print("Synthetic TTV quick-start")
    print("chi2 limits:", chi2_limit.tolist())
    print("rms limits:", rms_limit.tolist())
    print("sample megno:", float(megno))


if __name__ == "__main__":
    main()
