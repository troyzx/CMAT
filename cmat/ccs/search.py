"""One-model M0-M3 comparison; NOT a detection-significance calculator."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .data import TimingData
from .dynamics import companion_signals
from .likelihood import fit_reference_ephemeris, score_model


def compare_candidate(
    data: TimingData,
    *,
    star_mass_msun: float,
    inner_mass_mjup: float,
    outer_mass_mjup: float,
    outer_period_day: float,
    outer_phase_deg: float,
    ephemeris_mode: str = "linear",
    max_epoch_span: int = 3000,
) -> dict:
    """Fit four nested, *conditional* signal families at one physical point.

    Phase, mass, period, eccentricity, osculating P are NOT optimized. M0 uses
    exactly the same ephemeris and noise model as M1/M2/M3.
    """
    t0, period, e_ref = fit_reference_ephemeris(data)
    unique = np.unique(data.epoch)
    unique_signals = companion_signals(
        unique,
        star_mass_msun=star_mass_msun,
        inner_mass_mjup=inner_mass_mjup,
        inner_period_day=period,
        outer_mass_mjup=outer_mass_mjup,
        outer_period_day=outer_period_day,
        outer_phase_deg=outer_phase_deg,
        max_epoch_span=max_epoch_span,
    )
    idx = np.searchsorted(unique, data.epoch)
    results = {}
    for label, unique_signal in unique_signals.items():
        signal = unique_signal[idx]
        fit = score_model(
            data, signal,
            reference_t0_bjd=t0, reference_period_day=period,
            reference_epoch=e_ref, mode=ephemeris_mode,
        )
        record = asdict(fit)
        record.pop("residual_sec")
        record["reduced_chi2"] = fit.reduced_chi2
        record["max_abs_signal_sec"] = float(np.max(np.abs(signal)))
        results[label] = record
    for label in ("M1", "M2", "M3"):
        results[label]["delta_chi2_vs_M0"] = (
            results["M0"]["chi2"] - results[label]["chi2"]
        )
    return dict(
        status="PILOT_CONDITIONAL_NOT_DETECTION",
        n_timings=data.n,
        n_unique_epochs=len(unique),
        timing_input_sha256=data.input_sha256,
        ephemeris_mode=ephemeris_mode,
        reference=dict(t0_bjd_tdb=t0, period_day=period, epoch=e_ref),
        hypothesis=dict(star_mass_msun=star_mass_msun,
                        inner_mass_mjup=inner_mass_mjup,
                        outer_mass_mjup=outer_mass_mjup,
                        outer_period_day=outer_period_day,
                        outer_phase_deg=outer_phase_deg,
                        eccentricity=0.0, coplanar=True, inclination_deg=90.0),
        scores=results,
        caveats=[
            "Only one companion point evaluated; search and phase not optimized.",
            "Inner osculating period fixed to a weighted observed period.",
            "Ephemeris linear-profile is not a dynamical re-integration of P.",
            "No false-alarm correction, noise covariance, RV, or stability test.",
        ],
    )
