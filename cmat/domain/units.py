"""Import-safe unit conversion helpers for CMAT domain logic."""

from __future__ import annotations

import numpy as np

MJ_TO_MS = 9.5e-4
ME_TO_MS = 3.0e-6
RS_TO_AU = 0.00465047


def jupiter_mass_to_solar_mass(m_jupiter):
    """Convert Jupiter masses to solar masses."""

    return np.asarray(m_jupiter, dtype=float) * MJ_TO_MS


def earth_mass_to_solar_mass(m_earth):
    """Convert Earth masses to solar masses."""

    return np.asarray(m_earth, dtype=float) * ME_TO_MS


def solar_radius_to_au(r_solar):
    """Convert solar radii to astronomical units."""

    return np.asarray(r_solar, dtype=float) * RS_TO_AU
