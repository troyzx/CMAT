"""Workflow adapters built around CMAT configuration objects.

The functions in this module delegate to the existing fitting and simulation
classes. They provide a stable library boundary for notebooks, future CLI
entry points, and provenance capture without changing the scientific
implementation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

from .config import RunConfig, TargetConfig


DEFAULT_PROVENANCE_DISTRIBUTIONS = (
    "CMAT-astro",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "tqdm",
    "requests",
    "astroquery",
    "pytransit",
    "arviz",
    "celerite",
    "emcee",
    "corner",
    "rebound",
    "uncertainties",
    "numba",
    "llvmlite",
)


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
    scoring_backend=None,
):
    """Create a `TTVSimulation` instance from a run configuration and TTV data."""

    if config.simulation is None:
        raise ValueError("config.simulation is required to create a TTV simulation")

    if scoring_backend is None:
        from .scoring import make_mass_threshold_scorer

        scoring_backend = make_mass_threshold_scorer(
            config.scoring.backend,
            bayesian_config=config.scoring.bayesian,
        )

    from .ttv_sim import TTVSimulation

    simulation = TTVSimulation(
        epochs=epochs,
        ttv_mcmc=ttv_mcmc,
        ttv_err=ttv_err,
        rs=config.simulation.period_ratios,
        mp2s=config.simulation.companion_masses,
        prop=prop,
        N=config.simulation.n_transit_simulations,
        scoring_backend=scoring_backend,
    )
    simulation.megno_dt = config.simulation.megno_dt
    simulation.megno_runtime = config.simulation.megno_runtime
    simulation.worker_count = config.execution.worker_count
    simulation.start_method = config.execution.start_method
    simulation.show_progress = config.execution.show_progress
    return simulation


def workflow_manifest(
    config: RunConfig,
    *,
    dependency_versions: dict[str, str] | None = None,
    notes: dict[str, Any] | None = None,
    scoring_summary: Any | None = None,
    code_version: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a configured workflow run."""

    manifest: dict[str, Any] = {"config": config.to_dict()}
    if code_version is not None:
        manifest["code_version"] = code_version
    if dependency_versions is not None:
        manifest["dependency_versions"] = dict(sorted(dependency_versions.items()))
    if runtime is not None:
        manifest["runtime"] = runtime
    if notes is not None:
        manifest["notes"] = notes
    if scoring_summary is not None:
        if hasattr(scoring_summary, "to_dict"):
            scoring_summary = scoring_summary.to_dict()
        manifest["scoring_summary"] = scoring_summary
    return manifest


def provenance_dependency_versions(
    distributions: tuple[str, ...] = DEFAULT_PROVENANCE_DISTRIBUTIONS,
) -> dict[str, str]:
    """Return installed versions for the provenance dependency surface."""

    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            continue
    return dict(sorted(versions.items()))


def provenance_code_version() -> dict[str, Any]:
    """Return git-based source provenance when available."""

    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def provenance_runtime() -> dict[str, Any]:
    """Return runtime context for a persisted workflow artifact."""

    environment = {
        name: value
        for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME")
        if (value := os.environ.get(name)) is not None
    }
    runtime: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    if environment:
        runtime["environment"] = environment
    return runtime


def write_workflow_manifest(
    config: RunConfig,
    *,
    dependency_versions: dict[str, str] | None = None,
    notes: dict[str, Any] | None = None,
    scoring_summary: Any | None = None,
    code_version: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    metadata_path: str | Path | None = None,
) -> Path:
    """Persist a workflow manifest to the configured metadata path."""

    if metadata_path is None:
        metadata_path = config.output.metadata_path
    metadata_path = Path(metadata_path)
    manifest = workflow_manifest(
        config,
        dependency_versions=(
            provenance_dependency_versions()
            if dependency_versions is None
            else dependency_versions
        ),
        notes=notes,
        scoring_summary=scoring_summary,
        code_version=provenance_code_version() if code_version is None else code_version,
        runtime=provenance_runtime() if runtime is None else runtime,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _require_simulation_grid(config: RunConfig):
    if config.simulation is None:
        raise ValueError("config.simulation is required for simulation cache helpers")
    return config.simulation


def _resolve_cache_path(default_path: Path, explicit_path: str | Path | None) -> Path:
    return default_path if explicit_path is None else Path(explicit_path)


def write_ttv_grid_cache(
    config: RunConfig,
    *,
    epochs,
    ttv_mcmc,
    ttv_err,
    ttv_results,
    cache_path: str | Path | None = None,
) -> Path:
    """Persist a reusable TTV grid cache for a configured simulation run."""

    simulation = _require_simulation_grid(config)
    ttv_results = np.asarray(ttv_results, dtype=float)
    if ttv_results.ndim != 2:
        raise ValueError("ttv_results must be a two-dimensional array-like")
    if ttv_results.shape[0] != simulation.parameter_count:
        raise ValueError("ttv_results must match the configured simulation grid size")

    epochs = np.asarray(epochs, dtype=int)
    ttv_mcmc = np.asarray(ttv_mcmc, dtype=float)
    ttv_err = np.asarray(ttv_err, dtype=float)
    if ttv_mcmc.shape != ttv_err.shape:
        raise ValueError("ttv_mcmc and ttv_err must have matching shapes")

    cache_path = _resolve_cache_path(config.output.ttv_grid_cache_path, cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            period_ratios=simulation.period_ratios,
            companion_masses=simulation.companion_masses,
            epochs=epochs,
            ttv_mcmc=ttv_mcmc,
            ttv_err=ttv_err,
            ttv_results=ttv_results,
        )
    return cache_path


def load_ttv_grid_cache(
    config: RunConfig,
    *,
    cache_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load a cached TTV grid written by `write_ttv_grid_cache(...)`."""

    cache_path = _resolve_cache_path(config.output.ttv_grid_cache_path, cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def write_megno_grid_cache(
    config: RunConfig,
    *,
    megno_results,
    cache_path: str | Path | None = None,
) -> Path:
    """Persist a reusable MEGNO grid cache for a configured simulation run."""

    simulation = _require_simulation_grid(config)
    megno_results = np.asarray(megno_results, dtype=float)
    if megno_results.ndim != 1:
        raise ValueError("megno_results must be a one-dimensional array-like")
    if megno_results.size != simulation.parameter_count:
        raise ValueError("megno_results must match the configured simulation grid size")

    cache_path = _resolve_cache_path(config.output.megno_grid_cache_path, cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            period_ratios=simulation.period_ratios,
            companion_masses=simulation.companion_masses,
            megno_results=megno_results,
        )
    return cache_path


def load_megno_grid_cache(
    config: RunConfig,
    *,
    cache_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Load a cached MEGNO grid written by `write_megno_grid_cache(...)`."""

    cache_path = _resolve_cache_path(config.output.megno_grid_cache_path, cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _normalize_scoring_summary(scoring_summary: Any) -> dict[str, Any]:
    if hasattr(scoring_summary, "to_dict"):
        scoring_summary = scoring_summary.to_dict()
    if not isinstance(scoring_summary, dict):
        raise TypeError("scoring_summary must be a dict-like object or expose to_dict()")
    return scoring_summary


def _bayesian_cache_payload(scoring_summary: Any) -> dict[str, Any]:
    payload = _normalize_scoring_summary(scoring_summary)
    bayesian = payload.get("bayesian", payload)
    if not isinstance(bayesian, dict):
        raise ValueError("scoring_summary does not contain a Bayesian payload")
    posterior_samples = bayesian.get("posterior_samples")
    if posterior_samples is None:
        raise ValueError("scoring_summary does not contain posterior_samples")

    cache_payload = {
        key: bayesian[key]
        for key in (
            "status",
            "contract_version",
            "sampler",
            "credible_interval",
            "observed_transit_count",
            "sample_count",
            "requested_sample_count",
            "warmup_draws",
            "nuisance_parameters",
            "reference_solution",
            "diagnostics",
            "posterior_samples",
        )
        if key in bayesian
    }
    cache_payload["posterior_samples"] = {
        name: [float(value) for value in values]
        for name, values in posterior_samples.items()
    }
    return cache_payload


def write_posterior_samples_cache(
    config: RunConfig,
    *,
    scoring_summary: Any,
    cache_path: str | Path | None = None,
) -> Path:
    """Persist retained Bayesian posterior samples for reuse."""

    cache_path = _resolve_cache_path(config.output.posterior_samples_cache_path, cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(_bayesian_cache_payload(scoring_summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cache_path


def load_posterior_samples_cache(
    config: RunConfig,
    *,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load retained Bayesian posterior samples written by `write_posterior_samples_cache(...)`."""

    cache_path = _resolve_cache_path(config.output.posterior_samples_cache_path, cache_path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["posterior_samples"] = {
        name: np.asarray(values, dtype=float)
        for name, values in payload["posterior_samples"].items()
    }
    return payload
