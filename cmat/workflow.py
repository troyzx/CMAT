"""Workflow adapters built around CMAT configuration objects.

The functions in this module delegate to the existing fitting and simulation
classes. They provide a stable library boundary for notebooks, future CLI
entry points, and provenance capture without changing the scientific
implementation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from multiprocessing import get_context
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Protocol, Sequence

import numpy as np
from tqdm.auto import tqdm

from .config import ExecutionConfig, RunConfig, TargetConfig


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


@dataclass(frozen=True)
class SimulatorAdapterRunResult:
    """Structured result from a generic simulator-adapter evaluation."""

    parameter_grid: tuple[dict[str, float], ...]
    scores: np.ndarray
    accepted: np.ndarray
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_grid": [dict(parameters) for parameters in self.parameter_grid],
            "scores": self.scores.tolist(),
            "accepted": self.accepted.tolist(),
            "summary": self.summary,
        }


class SimulatorAdapter(Protocol):
    """Protocol for non-astronomy simulator adapters used by the Stage 5 workflow."""

    def parameter_grid(self) -> Sequence[dict[str, float]]: ...

    def simulate(self, parameters: dict[str, float]) -> Any: ...

    def score(self, simulated_observable: Any) -> float: ...

    def is_accepted(self, score: float) -> bool: ...

    def summarize(
        self,
        *,
        parameter_grid: tuple[dict[str, float], ...],
        scores: np.ndarray,
        accepted: np.ndarray,
    ) -> dict[str, Any]: ...


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


def _normalize_parameter_grid(
    parameter_grid: Sequence[dict[str, float]],
) -> tuple[dict[str, float], ...]:
    normalized: list[dict[str, float]] = []
    for parameters in parameter_grid:
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("adapter parameter_grid() must return non-empty dictionaries")
        normalized_parameters: dict[str, float] = {}
        for name, value in parameters.items():
            if not isinstance(name, str):
                raise TypeError("adapter parameter names must be strings")
            name = name.strip()
            if not name:
                raise ValueError("adapter parameter names must not be empty")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("adapter parameter values must be finite")
            normalized_parameters[name] = value
        normalized.append(dict(sorted(normalized_parameters.items())))
    if not normalized:
        raise ValueError("adapter parameter_grid() must not be empty")
    return tuple(normalized)


def _adapter_score_task(task: tuple[SimulatorAdapter, dict[str, float]]) -> float:
    adapter, parameters = task
    return float(adapter.score(adapter.simulate(parameters)))


def run_simulator_adapter(
    adapter: SimulatorAdapter,
    *,
    execution: ExecutionConfig | None = None,
) -> SimulatorAdapterRunResult:
    """Run a generic simulator adapter over its latent-state grid.

    This Stage 5 helper keeps the astronomy-specific TTV path intact while
    exposing the same inverse-modeling workflow shape for other physical
    simulation domains.
    """

    if execution is None:
        execution = ExecutionConfig(show_progress=False)
    if not isinstance(execution, ExecutionConfig):
        raise TypeError("execution must be an ExecutionConfig or None")

    parameter_grid = _normalize_parameter_grid(adapter.parameter_grid())
    tasks = [(adapter, parameters) for parameters in parameter_grid]

    if execution.worker_count == 1:
        iterator = map(_adapter_score_task, tasks)
    else:
        context = get_context(execution.start_method)
        pool = context.Pool(execution.worker_count)
        try:
            iterator = pool.imap(_adapter_score_task, tasks)
            if execution.show_progress:
                iterator = tqdm(iterator, total=len(tasks))
            scores = np.asarray(list(iterator), dtype=float)
        finally:
            pool.close()
            pool.join()
        if not np.all(np.isfinite(scores)):
            raise ValueError("adapter scores must be finite")
        accepted = np.asarray(
            [bool(adapter.is_accepted(float(score))) for score in scores],
            dtype=bool,
        )
        return SimulatorAdapterRunResult(
            parameter_grid=parameter_grid,
            scores=scores,
            accepted=accepted,
            summary=adapter.summarize(
                parameter_grid=parameter_grid,
                scores=scores.copy(),
                accepted=accepted.copy(),
            ),
        )

    if execution.show_progress:
        iterator = tqdm(iterator, total=len(tasks))
    scores = np.asarray(list(iterator), dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("adapter scores must be finite")
    accepted = np.asarray(
        [bool(adapter.is_accepted(float(score))) for score in scores],
        dtype=bool,
    )
    return SimulatorAdapterRunResult(
        parameter_grid=parameter_grid,
        scores=scores,
        accepted=accepted,
        summary=adapter.summarize(
            parameter_grid=parameter_grid,
            scores=scores.copy(),
            accepted=accepted.copy(),
        ),
    )


def _write_simulator_adapter_figure(
    result: SimulatorAdapterRunResult,
    *,
    figures_dir: Path,
    parameter_names: tuple[str, ...],
    title: str,
) -> Path | None:
    if len(parameter_names) not in (1, 2):
        return None

    from matplotlib import pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "score_surface.png"
    fig, ax = plt.subplots(figsize=(7, 5))

    if len(parameter_names) == 1:
        x_name = parameter_names[0]
        x = np.asarray([parameters[x_name] for parameters in result.parameter_grid], dtype=float)
        ax.plot(x, result.scores, marker="o", color="tab:blue")
        accepted_mask = np.asarray(result.accepted, dtype=bool)
        if np.any(accepted_mask):
            ax.scatter(
                x[accepted_mask],
                result.scores[accepted_mask],
                color="tab:green",
                label="accepted",
                zorder=3,
            )
        ax.set_xlabel(x_name)
        ax.set_ylabel("score")
    else:
        x_name, y_name = parameter_names
        x = np.asarray([parameters[x_name] for parameters in result.parameter_grid], dtype=float)
        y = np.asarray([parameters[y_name] for parameters in result.parameter_grid], dtype=float)
        scatter = ax.scatter(x, y, c=result.scores, cmap="viridis", s=120)
        accepted_mask = np.asarray(result.accepted, dtype=bool)
        if np.any(accepted_mask):
            ax.scatter(
                x[accepted_mask],
                y[accepted_mask],
                facecolors="none",
                edgecolors="white",
                linewidths=2.0,
                s=180,
                label="accepted",
            )
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("score")

    ax.set_title(title)
    ax.grid(alpha=0.25)
    if np.any(result.accepted):
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def write_simulator_adapter_portfolio(
    result: SimulatorAdapterRunResult,
    *,
    output_dir: str | Path,
    title: str = "Simulator adapter portfolio output",
) -> dict[str, Path]:
    """Write report, table, and figure artifacts for a simulator-adapter run."""

    if not isinstance(result, SimulatorAdapterRunResult):
        raise TypeError("result must be a SimulatorAdapterRunResult")

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    table_path = tables_dir / "grid_scores.csv"

    parameter_names = tuple(result.parameter_grid[0].keys()) if result.parameter_grid else ()
    candidate_count = len(result.parameter_grid)
    accepted_count = int(np.sum(result.accepted))
    best_index = int(np.argmin(result.scores))
    best_parameters = result.parameter_grid[best_index]
    best_score = float(result.scores[best_index])

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "title": title,
        "parameter_names": list(parameter_names),
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "best_parameters": best_parameters,
        "best_score": best_score,
        "result_summary": result.summary,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*parameter_names, "score", "accepted"],
        )
        writer.writeheader()
        for parameters, score, accepted in zip(
            result.parameter_grid,
            result.scores,
            result.accepted,
            strict=True,
        ):
            row = dict(parameters)
            row["score"] = float(score)
            row["accepted"] = bool(accepted)
            writer.writerow(row)

    report_path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "## Summary",
                "",
                f"- candidates evaluated: {candidate_count}",
                f"- accepted candidates: {accepted_count}",
                f"- best score: {best_score:.6f}",
                f"- best parameters: `{json.dumps(best_parameters, sort_keys=True)}`",
                "",
                "## Artifact paths",
                "",
                f"- summary JSON: `{summary_path}`",
                f"- score table CSV: `{table_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paths = {
        "report": report_path,
        "summary": summary_path,
        "table": table_path,
    }
    figure_path = _write_simulator_adapter_figure(
        result,
        figures_dir=figures_dir,
        parameter_names=parameter_names,
        title=title,
    )
    if figure_path is not None:
        report_path.write_text(
            report_path.read_text(encoding="utf-8")
            + f"- score figure: `{figure_path}`\n",
            encoding="utf-8",
        )
        paths["figure"] = figure_path
    return paths


def simulator_adapter_manifest(
    result: SimulatorAdapterRunResult,
    *,
    adapter_name: str,
    execution: ExecutionConfig | None = None,
    dependency_versions: dict[str, str] | None = None,
    notes: dict[str, Any] | None = None,
    code_version: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a simulator-adapter run."""

    if not isinstance(result, SimulatorAdapterRunResult):
        raise TypeError("result must be a SimulatorAdapterRunResult")
    if not isinstance(adapter_name, str) or not adapter_name.strip():
        raise ValueError("adapter_name must be a non-empty string")
    if execution is not None and not isinstance(execution, ExecutionConfig):
        raise TypeError("execution must be an ExecutionConfig or None")

    parameter_names = tuple(result.parameter_grid[0].keys()) if result.parameter_grid else ()
    accepted_count = int(np.sum(result.accepted))
    best_index = int(np.argmin(result.scores))

    manifest: dict[str, Any] = {
        "adapter_name": adapter_name.strip(),
        "parameter_names": list(parameter_names),
        "candidate_count": len(result.parameter_grid),
        "accepted_count": accepted_count,
        "best_parameters": result.parameter_grid[best_index],
        "best_score": float(result.scores[best_index]),
        "result_summary": result.summary,
    }
    if execution is not None:
        manifest["execution"] = execution.to_dict()
    if dependency_versions is not None:
        manifest["dependency_versions"] = dict(sorted(dependency_versions.items()))
    if code_version is not None:
        manifest["code_version"] = code_version
    if runtime is not None:
        manifest["runtime"] = runtime
    if notes is not None:
        manifest["notes"] = notes
    return manifest


def write_simulator_adapter_manifest(
    result: SimulatorAdapterRunResult,
    *,
    adapter_name: str,
    metadata_path: str | Path,
    execution: ExecutionConfig | None = None,
    dependency_versions: dict[str, str] | None = None,
    notes: dict[str, Any] | None = None,
    code_version: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> Path:
    """Persist a simulator-adapter run manifest for deployment-style examples."""

    metadata_path = Path(metadata_path)
    manifest = simulator_adapter_manifest(
        result,
        adapter_name=adapter_name,
        execution=execution,
        dependency_versions=(
            provenance_dependency_versions()
            if dependency_versions is None
            else dependency_versions
        ),
        notes=notes,
        code_version=provenance_code_version() if code_version is None else code_version,
        runtime=provenance_runtime() if runtime is None else runtime,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


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
    if epochs.ndim != 1 or ttv_mcmc.ndim != 1 or ttv_err.ndim != 1:
        raise ValueError("epochs, ttv_mcmc, and ttv_err must be one-dimensional")
    if epochs.shape != ttv_mcmc.shape or ttv_mcmc.shape != ttv_err.shape:
        raise ValueError("epochs, ttv_mcmc, and ttv_err must have matching shapes")

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
            "rejection_log_bayes_factor_threshold",
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
