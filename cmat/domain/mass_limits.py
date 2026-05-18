"""Pure mass-threshold helpers for score surfaces."""

from __future__ import annotations

import numpy as np


def first_rejected_masses(
    score_surface,
    masses,
    threshold,
    *,
    reject_if="greater_equal",
    valid_mask=None,
):
    """Return the first rejected mass in each score-surface column."""

    score_surface = np.asarray(score_surface, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if score_surface.ndim != 2:
        raise ValueError("score_surface must be two-dimensional")
    if score_surface.shape[0] != len(masses):
        raise ValueError("score_surface row count must match masses")

    if valid_mask is None:
        valid_mask = np.isfinite(score_surface)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != score_surface.shape:
            raise ValueError("valid_mask must match score_surface shape")

    if reject_if == "greater_equal":

        def reject(value):
            return value >= threshold
    else:
        raise ValueError(f"Unsupported reject_if rule: {reject_if}")

    rejected_masses = []
    for column_index in range(score_surface.shape[1]):
        for mass_index, mass in enumerate(masses):
            if not valid_mask[mass_index, column_index]:
                continue
            value = score_surface[mass_index, column_index]
            if not np.isfinite(value):
                continue
            if reject(value):
                rejected_masses.append(mass)
                break

    return np.asarray(rejected_masses, dtype=float)
