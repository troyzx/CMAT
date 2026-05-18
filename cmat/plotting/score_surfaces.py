"""Plotting helpers for chi-squared-style score surfaces."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def plot_score_surface(
    *,
    period_ratios,
    companion_masses,
    surface,
    statistic_label,
    vmin=None,
    vmax=None,
    threshold=None,
    show_threshold=True,
    threshold_color="white",
    levels=None,
    ax=None,
    cmap="plasma",
    figsize=(4, 3.2),
    dpi=200,
):
    """Plot a score surface over companion mass and period-ratio grids."""

    period_ratios = np.asarray(period_ratios, dtype=float)
    companion_masses = np.asarray(companion_masses, dtype=float)
    surface = np.asarray(surface, dtype=float)

    if surface.shape != (len(companion_masses), len(period_ratios)):
        raise ValueError("score surface shape must match the configured mp2-r grid")
    if len(period_ratios) < 2 or len(companion_masses) < 2:
        raise ValueError("contour plotting requires at least a 2x2 mp2-r grid")
    if not np.any(np.isfinite(surface)):
        raise ValueError("score surface must contain at least one finite value")
    finite_surface = surface[np.isfinite(surface)]
    surface_varies = not np.allclose(finite_surface, finite_surface[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    masked_surface = np.ma.masked_invalid(surface)
    x_grid, y_grid = np.meshgrid(period_ratios, companion_masses)
    mesh = ax.pcolor(
        x_grid,
        y_grid,
        masked_surface,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(statistic_label)

    if (
        show_threshold
        and threshold is not None
        and np.isfinite(threshold)
        and surface_varies
        and np.nanmin(surface) <= threshold <= np.nanmax(surface)
    ):
        threshold_contour = ax.contour(
            period_ratios,
            companion_masses,
            masked_surface,
            levels=[threshold],
            colors=threshold_color,
            linewidths=1.2,
        )
        ax.clabel(threshold_contour, fmt={threshold: "limit"})

    ax.set_xlabel(r"$P_2/P_1$")
    ax.set_ylabel(r"Mass [$M_\oplus$]")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    return fig, ax
