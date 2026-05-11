"""Typed configuration objects for CMAT workflows.

These dataclasses make workflow inputs explicit without executing the fitting
or simulation stack. They are intentionally small and JSON-serializable so the
same contract can be reused by notebooks, command-line entry points, tests,
and batch runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return int(value)


def _require_nonnegative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return int(value)


def _require_positive_float(value: float, field_name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _as_1d_float_array(
    values: Iterable[float] | np.ndarray,
    field_name: str,
    *,
    strictly_increasing: bool = False,
) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=float)
    else:
        array = np.asarray(list(values), dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    if np.any(array <= 0):
        raise ValueError(f"{field_name} values must be positive")
    if strictly_increasing and np.any(np.diff(array) <= 0):
        raise ValueError(f"{field_name} must be strictly increasing")
    return array


@dataclass(frozen=True)
class TargetConfig:
    """Target-level inputs for data discovery and local storage."""

    planet_name: str
    data_dir: Path | str = Path("data")
    product_subgroups: tuple[str, ...] = ("LC",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "planet_name", _require_text(self.planet_name, "planet_name"))
        object.__setattr__(self, "data_dir", Path(self.data_dir))

        if isinstance(self.product_subgroups, str):
            products = (self.product_subgroups,)
        else:
            products = tuple(self.product_subgroups)
        if not products:
            raise ValueError("product_subgroups must contain at least one value")
        products = tuple(_require_text(product, "product_subgroups") for product in products)
        object.__setattr__(self, "product_subgroups", products)

    @property
    def target_data_dir(self) -> Path:
        return self.data_dir / self.planet_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "planet_name": self.planet_name,
            "data_dir": str(self.data_dir),
            "target_data_dir": str(self.target_data_dir),
            "product_subgroups": list(self.product_subgroups),
        }


@dataclass(frozen=True)
class FitControls:
    """Optimizer and sampler controls for the light-curve fitting stages."""

    global_niter: int = 200
    global_npop: int = 30
    single_niter: int = 100
    single_npop: int = 50
    mcmc_steps: int = 2500
    mcmc_thin: int = 25
    mcmc_repeats: int = 4

    def __post_init__(self) -> None:
        for field_name in (
            "global_niter",
            "global_npop",
            "single_niter",
            "single_npop",
            "mcmc_steps",
            "mcmc_thin",
            "mcmc_repeats",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_positive_int(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "global_niter": self.global_niter,
            "global_npop": self.global_npop,
            "single_niter": self.single_niter,
            "single_npop": self.single_npop,
            "mcmc_steps": self.mcmc_steps,
            "mcmc_thin": self.mcmc_thin,
            "mcmc_repeats": self.mcmc_repeats,
        }


@dataclass(frozen=True)
class SimulationGrid:
    """Forward-simulation grid for companion period ratios and masses."""

    period_ratios: Iterable[float] | np.ndarray
    companion_masses: Iterable[float] | np.ndarray
    n_transit_simulations: int = 80
    megno_dt: float = 1 / 20
    megno_runtime: float = 1e4

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "period_ratios",
            _as_1d_float_array(self.period_ratios, "period_ratios"),
        )
        object.__setattr__(
            self,
            "companion_masses",
            _as_1d_float_array(
                self.companion_masses,
                "companion_masses",
                strictly_increasing=True,
            ),
        )
        object.__setattr__(
            self,
            "n_transit_simulations",
            _require_positive_int(self.n_transit_simulations, "n_transit_simulations"),
        )
        object.__setattr__(self, "megno_dt", _require_positive_float(self.megno_dt, "megno_dt"))
        object.__setattr__(
            self,
            "megno_runtime",
            _require_positive_float(self.megno_runtime, "megno_runtime"),
        )

    @property
    def parameter_count(self) -> int:
        return len(self.period_ratios) * len(self.companion_masses)

    def parameter_pairs(self) -> list[tuple[float, float]]:
        return [
            (float(period_ratio), float(companion_mass))
            for companion_mass in self.companion_masses
            for period_ratio in self.period_ratios
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_ratios": self.period_ratios.tolist(),
            "companion_masses": self.companion_masses.tolist(),
            "n_transit_simulations": self.n_transit_simulations,
            "megno_dt": self.megno_dt,
            "megno_runtime": self.megno_runtime,
            "parameter_count": self.parameter_count,
        }


@dataclass(frozen=True)
class OutputConfig:
    """Filesystem destinations for run artifacts."""

    root_dir: Path | str = Path("artifacts")
    run_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_dir", Path(self.root_dir))
        if self.run_name is not None:
            object.__setattr__(self, "run_name", _require_text(self.run_name, "run_name"))

    @property
    def run_dir(self) -> Path:
        if self.run_name is None:
            return self.root_dir
        return self.root_dir / self.run_name

    @property
    def tables_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "run_metadata.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "tables_dir": str(self.tables_dir),
            "figures_dir": str(self.figures_dir),
            "metadata_path": str(self.metadata_path),
        }


@dataclass(frozen=True)
class ScoringConfig:
    """Typed selector for the current mass-threshold scoring backend."""

    backend: str = "chi2_rms"

    def __post_init__(self) -> None:
        from .scoring import supported_mass_threshold_backends

        backend = _require_text(self.backend, "backend")
        if backend not in supported_mass_threshold_backends():
            raise ValueError(
                "backend must be one of "
                + ", ".join(supported_mass_threshold_backends())
            )
        object.__setattr__(self, "backend", backend)

    def to_dict(self) -> dict[str, str]:
        return {"backend": self.backend}


@dataclass(frozen=True)
class RunConfig:
    """Composite configuration for a CMAT fitting and simulation run."""

    target: TargetConfig
    fit: FitControls = field(default_factory=FitControls)
    simulation: SimulationGrid | None = None
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, TargetConfig):
            raise TypeError("target must be a TargetConfig")
        if not isinstance(self.fit, FitControls):
            raise TypeError("fit must be a FitControls")
        if self.simulation is not None and not isinstance(self.simulation, SimulationGrid):
            raise TypeError("simulation must be a SimulationGrid or None")
        if not isinstance(self.scoring, ScoringConfig):
            raise TypeError("scoring must be a ScoringConfig")
        if not isinstance(self.output, OutputConfig):
            raise TypeError("output must be an OutputConfig")
        if self.random_seed is not None:
            object.__setattr__(
                self,
                "random_seed",
                _require_nonnegative_int(self.random_seed, "random_seed"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "fit": self.fit.to_dict(),
            "simulation": None if self.simulation is None else self.simulation.to_dict(),
            "scoring": self.scoring.to_dict(),
            "output": self.output.to_dict(),
            "random_seed": self.random_seed,
        }
