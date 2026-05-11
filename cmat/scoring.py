"""Scoring helpers and backend interfaces for comparing simulated TTV grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.stats


def get_chi2(ttv_rebound, epoch, ttv_mcmc, ttv_err):
    """Return the best chi-squared score over possible epoch alignments."""

    rangea = range(epoch[-1] - epoch[0])
    T = [ttv_rebound[np.array(epoch - epoch[0]) + a] for a in rangea]
    chi2 = (((T - ttv_mcmc) ** 2) / ttv_err**2).sum(axis=1)
    return chi2.min()


def get_rms(ttv_rebound):
    """Return the root-mean-square amplitude of simulated TTV residuals."""

    rms = np.sqrt(np.mean(ttv_rebound**2))
    return rms


@dataclass(frozen=True)
class MassThresholds:
    """Critical-mass curves derived from one scoring backend over a TTV grid."""

    chi2: np.ndarray
    rms: np.ndarray
    backend: str = "chi2_rms"
    chi2_threshold: float | None = None
    rms_threshold: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "chi2": self.chi2.tolist(),
            "rms": self.rms.tolist(),
            "chi2_threshold": self.chi2_threshold,
            "rms_threshold": self.rms_threshold,
        }


class MassThresholdScorer(Protocol):
    """Protocol for backend objects that extract critical-mass curves."""

    def critical_masses(
        self,
        *,
        ttv_results,
        epoch,
        ttv_mcmc,
        ttv_err,
        period_ratios,
        companion_masses,
    ) -> MassThresholds: ...


def first_rejected_mass(score_2d, crit, *, valid_2d, period_ratios, companion_masses):
    """Return the first rejected companion mass for each period-ratio column."""

    rejected_masses = []
    for ratio_index, _ in enumerate(period_ratios):
        for mass_index, companion_mass in enumerate(companion_masses):
            if not valid_2d[mass_index, ratio_index]:
                continue
            if not np.isfinite(score_2d[mass_index, ratio_index]):
                continue
            if score_2d[mass_index, ratio_index] >= crit:
                rejected_masses.append(companion_mass)
                break
    return np.array(rejected_masses)


class Chi2AndRmsMassThresholdScorer:
    """Default scoring backend that preserves the current chi2/RMS behavior."""

    def critical_masses(
        self,
        *,
        ttv_results,
        epoch,
        ttv_mcmc,
        ttv_err,
        period_ratios,
        companion_masses,
    ) -> MassThresholds:
        chi2 = get_chi2_v(
            ttv_rebound=np.array(ttv_results),
            epoch=epoch,
            ttv_mcmc=ttv_mcmc,
            ttv_err=ttv_err,
        )
        chi2_crit = scipy.stats.chi2.ppf(0.997, len(ttv_mcmc))

        rms = get_rms_v(ttv_results)
        rms_crit = np.sqrt(np.mean(ttv_mcmc**2))

        chi2_2d = np.array(chi2).reshape(len(companion_masses), len(period_ratios))
        rms_2d = np.array(rms).reshape(len(companion_masses), len(period_ratios))
        valid_2d = rms_2d != 0

        return MassThresholds(
            chi2=first_rejected_mass(
                chi2_2d,
                chi2_crit,
                valid_2d=valid_2d,
                period_ratios=period_ratios,
                companion_masses=companion_masses,
            ),
            rms=first_rejected_mass(
                rms_2d,
                rms_crit,
                valid_2d=valid_2d,
                period_ratios=period_ratios,
                companion_masses=companion_masses,
            ),
            backend="chi2_rms",
            chi2_threshold=chi2_crit,
            rms_threshold=rms_crit,
        )


def make_mass_threshold_scorer(backend: str) -> MassThresholdScorer:
    """Build a supported mass-threshold scorer from a typed backend name."""

    if backend == "chi2_rms":
        return Chi2AndRmsMassThresholdScorer()
    raise ValueError(f"Unsupported scoring backend: {backend}")


get_chi2_v = np.vectorize(
    get_chi2,
    excluded=["epoch", "ttv_mcmc", "ttv_err"],
    signature="(n)->()",
)
get_rms_v = np.vectorize(get_rms, signature="(n)->()")
