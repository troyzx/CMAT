"""No-REBOUND synthetic sanity demo for the profiled-ephemeris likelihood."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .data import TimingData
from .likelihood import fit_reference_ephemeris, score_model


def synthetic_profile_check() -> dict:
    epochs = np.array([0, 1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 200])
    base_t0 = 2459000.0
    base_period = 1.4
    true_signal = 1.2*np.sin(2*np.pi*epochs/37.0+0.6)
    # Choose a deterministic reproducible profile challenge, not noisy data.
    simulated_shift = 0.45 + 0.003*epochs
    observed = base_t0 + base_period*epochs + (
        true_signal+simulated_shift
    ) / 86400.0
    data = TimingData(epochs, observed, np.full(len(epochs), 0.2))
    reference_t0, reference_period, ref_epoch = fit_reference_ephemeris(data)
    fits = {
        mode: score_model(data, true_signal,
                          reference_t0_bjd=reference_t0,
                          reference_period_day=reference_period,
                          reference_epoch=ref_epoch, mode=mode)
        for mode in ("fixed", "offset", "linear")
    }
    assert fits["linear"].chi2 < 1e-3, "Synthetic profile did not recover signal"
    assert fits["fixed"].chi2 >= fits["offset"].chi2 >= fits["linear"].chi2
    return dict(
        status="SYNTHETIC_PROFILE_PASS",
        n=len(epochs),
        scores={mode: asdict(fit) | {"residual_sec": fit.residual_sec.tolist()}
                for mode, fit in fits.items()},
    )
