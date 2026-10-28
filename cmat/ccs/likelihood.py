"""CCS likelihood: profile linear ephemeris against raw event-center times.

Subtract a nearby absolute BJD reference before linear algebra to avoid
catastrophic errors from fitting 2.4-million-day absolute timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import TimingData

SECONDS_PER_DAY = 86400.0
Mode = Literal["fixed", "offset", "linear", "median"]


@dataclass(frozen=True)
class EphemerisFit:
    chi2: float
    n: int
    fitted_parameters: int
    offset_sec: float
    delta_period_sec: float
    residual_sec: np.ndarray

    @property
    def reduced_chi2(self) -> float:
        dof = self.n - self.fitted_parameters
        return self.chi2 / dof if dof > 0 else float("nan")


def _weighted_linear(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(
        np.column_stack((np.ones(len(x)), x)) / sigma[:, None],
        y / sigma,
        rcond=None,
    )[0]


def fit_reference_ephemeris(data: TimingData) -> tuple[float, float, int]:
    """Return best weighted (T_ref[BJD_TDB], P[day], E_ref), for conditioning.

    These are convenience reference coordinates, NOT external priors on the
    physical osculating period. Errors and systematics are not estimated here.
    """
    weighted_epoch = np.average(data.epoch, weights=1 / data.sigma_sec**2)
    ref_epoch = int(data.epoch[np.argmin(np.abs(data.epoch - weighted_epoch))])
    j = int(np.argmin(np.abs(data.epoch - ref_epoch)))
    nearby_time = float(data.time_bjd_tdb[j])
    x = data.epoch.astype(float) - ref_epoch
    y = (data.time_bjd_tdb - nearby_time) * SECONDS_PER_DAY
    beta = _weighted_linear(x, y, data.sigma_sec)
    return nearby_time + beta[0] / SECONDS_PER_DAY, beta[1] / SECONDS_PER_DAY, ref_epoch


def score_model(
    data: TimingData,
    model_signal_sec: np.ndarray,
    *,
    reference_t0_bjd: float,
    reference_period_day: float,
    reference_epoch: int,
    mode: Mode = "linear",
) -> EphemerisFit:
    """Compare an event-indexed *additive timing signal* to observations.

    The model signal is in SECONDS and equals simulated transit time minus
    initial linear ephemeris at identical epochs, optionally plus outer LTTE.
    Modes: fixed (0 nuisances), offset (weighted T0), linear (weighted T0/P),
    median (unweighted residual median, for historical WD1856 diagnostics).
    """
    if mode not in {"fixed", "offset", "linear", "median"}:
        raise ValueError(f"Unsupported ephemeris mode: {mode}")
    signal = np.asarray(model_signal_sec, dtype=float)
    if signal.shape != data.epoch.shape or not np.isfinite(signal).all():
        raise ValueError("Model signal must be finite and match observed epochs")
    if not np.isfinite(reference_t0_bjd) or not np.isfinite(reference_period_day):
        raise ValueError("Invalid reference ephemeris")
    if reference_period_day <= 0:
        raise ValueError("Reference period must be positive")
    x = data.epoch.astype(float) - reference_epoch
    obs_minus_linear = (
        (data.time_bjd_tdb - reference_t0_bjd) * SECONDS_PER_DAY
        - x * (reference_period_day * SECONDS_PER_DAY)
    )
    residual = obs_minus_linear - signal
    off = 0.0
    dp = 0.0
    params = 0
    if mode == "median":
        off = float(np.median(residual))
        params = 1
    elif mode == "offset":
        off = float(np.average(residual, weights=1 / data.sigma_sec**2))
        params = 1
    elif mode == "linear":
        beta = _weighted_linear(x, residual, data.sigma_sec)
        off, dp = float(beta[0]), float(beta[1])
        params = 2
    final = residual - off - dp * x
    return EphemerisFit(
        chi2=float(np.sum((final / data.sigma_sec) ** 2)),
        n=data.n, fitted_parameters=params,
        offset_sec=off, delta_period_sec=dp, residual_sec=final,
    )
