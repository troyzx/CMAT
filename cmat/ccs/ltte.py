"""Circular two-body outer-orbit light-time signal; analytic validation model."""
from __future__ import annotations

import numpy as np

G_AU3_MSUN_DAY2 = 0.0002959122082855911
C_AU_PER_DAY = 173.14463267424
SECONDS_PER_DAY = 86400.0
MJUP_TO_MSUN = 1.0 / 1047.348644  # IAU nominal mass ratio; use consistently.


def circular_ltte_seconds(
    time_day: np.ndarray,
    *,
    star_mass_msun: float,
    outer_mass_mjup: float,
    outer_period_day: float,
    phase_deg: float = 0.0,
    inclination_deg: float = 90.0,
    inner_mass_mjup: float = 0.0,
) -> np.ndarray:
    """Star's outer-orbit LTTE; phase origin is arbitrary for circular orbits.

    Outer Jacobi semimajor axis uses the inner star+planet barycentre as the
    primary. For a massive inner planet, the analytic expression is a useful
    two-orbit approximation; numerical CCS uses REBOUND stellar z directly.
    """
    t = np.asarray(time_day, dtype=float)
    if not np.isfinite(t).all():
        raise ValueError("Time must be finite")
    if star_mass_msun <= 0 or outer_mass_mjup < 0 or inner_mass_mjup < 0:
        raise ValueError("Masses must be physical")
    if not np.isfinite(outer_period_day) or outer_period_day <= 0:
        raise ValueError("Period must be positive")
    total = star_mass_msun + (inner_mass_mjup + outer_mass_mjup) * MJUP_TO_MSUN
    companion = outer_mass_mjup * MJUP_TO_MSUN
    a_au = (G_AU3_MSUN_DAY2 * total * (outer_period_day / (2*np.pi))**2)**(1 / 3)
    amplitude = (a_au * companion / total) * SECONDS_PER_DAY / C_AU_PER_DAY
    return amplitude * np.sin(np.deg2rad(inclination_deg)) * np.sin(
        np.deg2rad(phase_deg) + 2*np.pi*t/outer_period_day
    )
