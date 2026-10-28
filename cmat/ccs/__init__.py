"""CCS: experimental transit-clock search for outer companions.

This package is deliberately separate from legacy CMAT mass-limit routines.
No data downloads, simulations, or filesystem writes happen on import.
"""
from .data import TimingData, load_timings_csv
from .likelihood import EphemerisFit, fit_reference_ephemeris, score_model
from .ltte import circular_ltte_seconds

__all__ = [
    "TimingData", "load_timings_csv", "EphemerisFit",
    "fit_reference_ephemeris", "score_model", "circular_ltte_seconds",
]
