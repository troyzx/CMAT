"""Pure timing helpers for linear-ephemeris calculations."""

from __future__ import annotations

import numpy as np


def calculate_epochs(transit_times, zero_epoch, period):
    """Return integer epochs relative to a linear ephemeris."""

    transit_times = np.asarray(transit_times, dtype=float)
    return np.rint((transit_times - float(zero_epoch)) / float(period)).astype(int)


def linear_ephemeris(epoch, zero_epoch, period):
    """Evaluate a linear ephemeris at the requested epochs."""

    epoch = np.asarray(epoch, dtype=float)
    return float(zero_epoch) + epoch * float(period)


def timing_residuals(transit_times, epochs, zero_epoch, period):
    """Return observed-minus-linear-ephemeris timing residuals."""

    transit_times = np.asarray(transit_times, dtype=float)
    return transit_times - linear_ephemeris(epochs, zero_epoch, period)
