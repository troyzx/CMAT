"""Workflow adapters built around CMAT configuration objects.

The functions in this module delegate to the existing fitting and simulation
classes. They provide a stable library boundary for notebooks, future CLI
entry points, and provenance capture without changing the scientific
implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RunConfig, TargetConfig


def legacy_data_dir(target: TargetConfig) -> str:
    """Return a data-root string compatible with the current fitting workflow API."""

    data_dir = Path(target.data_dir).as_posix().rstrip("/")
    if data_dir == "":
        return "./"
    return f"{data_dir}/"


def make_fit_lpf(target: TargetConfig):
    """Create a `TransitFitWorkflow` instance from a target configuration."""

    from .base import TransitFitWorkflow

    return TransitFitWorkflow(target.planet_name, datadir=legacy_data_dir(target))


def make_ttv_simulation(
    config: RunConfig,
    *,
    epochs,
    ttv_mcmc,
    ttv_err,
    prop,
):
    """Create a `TTVSimulation` instance from a run configuration and TTV data."""

    if config.simulation is None:
        raise ValueError("config.simulation is required to create a TTV simulation")

    from .ttv_sim import TTVSimulation

    simulation = TTVSimulation(
        epochs=epochs,
        ttv_mcmc=ttv_mcmc,
        ttv_err=ttv_err,
        rs=config.simulation.period_ratios,
        mp2s=config.simulation.companion_masses,
        prop=prop,
        N=config.simulation.n_transit_simulations,
    )
    simulation.megno_dt = config.simulation.megno_dt
    simulation.megno_runtime = config.simulation.megno_runtime
    return simulation


def workflow_manifest(
    config: RunConfig,
    *,
    dependency_versions: dict[str, str] | None = None,
    notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a configured workflow run."""

    manifest: dict[str, Any] = {"config": config.to_dict()}
    if dependency_versions is not None:
        manifest["dependency_versions"] = dict(sorted(dependency_versions.items()))
    if notes is not None:
        manifest["notes"] = notes
    return manifest
