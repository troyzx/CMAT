"""Pure residual scoring helpers shared by legacy CMAT APIs."""

from __future__ import annotations

import numpy as np


def rms(values):
    """Return the root-mean-square amplitude of residual values."""

    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def chi2_with_epoch_shift(ttv_simulated, ttv_observed, ttv_errors, epochs):
    """Return the minimum chi-squared score over integer epoch shifts."""

    ttv_simulated = np.asarray(ttv_simulated, dtype=float)
    ttv_observed = np.asarray(ttv_observed, dtype=float)
    ttv_errors = np.asarray(ttv_errors, dtype=float)
    epochs = np.asarray(epochs, dtype=int)

    epoch_span = int(epochs[-1] - epochs[0])
    shift_count = len(ttv_simulated) - epoch_span
    if shift_count <= 0:
        raise ValueError("ttv_simulated must be long enough to support epoch alignment")

    relative_epochs = epochs - epochs[0]
    chi2_scores = []
    for shift_index in range(shift_count):
        aligned = ttv_simulated[relative_epochs + shift_index]
        chi2_scores.append(np.sum(((aligned - ttv_observed) ** 2) / ttv_errors**2))
    return float(np.min(chi2_scores))
