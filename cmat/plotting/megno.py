"""Plotting helpers for MEGNO surfaces."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def plot_megno_surface(
    *,
    period_ratios,
    companion_masses,
    megno_results,
    ax=None,
    cmap="RdYlGn_r",
):
    """Plot a MEGNO grid as a 2D color map."""

    period_ratios = np.asarray(period_ratios, dtype=float)
    companion_masses = np.asarray(companion_masses, dtype=float)
    results2d = np.asarray(megno_results, dtype=float).reshape(
        len(companion_masses), len(period_ratios)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    extent = [
        float(np.min(period_ratios)),
        float(np.max(period_ratios)),
        float(np.min(companion_masses)),
        float(np.max(companion_masses)),
    ]
    ax.set_xlim(extent[0], extent[1])
    ax.set_xlabel("$P_2/P_1$")
    ax.set_ylim(extent[2], extent[3])
    ax.set_ylabel("Mass [$M_j$]")
    image = ax.imshow(
        results2d,
        interpolation="none",
        vmin=1.9,
        vmax=10,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=extent,
        alpha=0.8,
    )
    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label("MEGNO $\\langle Y \\rangle$")
    ax.grid()
    ax.set_xticks([1.5, 2, 2.5, 3, 3.3, 3.5, 3.8, 4])
    ax.set_xlabel(r"$P_2/P_1$")
    ax.set_ylabel(r"$M_2$ [$M_j$]")
    return fig, ax
