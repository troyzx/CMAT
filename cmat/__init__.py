name = "fitransit"

__all__ = [
    "Fitlpf",
    "ttv_sim",
    "TargetConfig",
    "FitControls",
    "SimulationGrid",
    "OutputConfig",
    "RunConfig",
]


def __getattr__(attribute_name):
    if attribute_name == "Fitlpf":
        from .base import Fitlpf

        return Fitlpf
    if attribute_name == "ttv_sim":
        from .ttv_sim import ttv_sim

        return ttv_sim
    if attribute_name in {
        "TargetConfig",
        "FitControls",
        "SimulationGrid",
        "OutputConfig",
        "RunConfig",
    }:
        from .config import (
            FitControls,
            OutputConfig,
            RunConfig,
            SimulationGrid,
            TargetConfig,
        )

        return {
            "TargetConfig": TargetConfig,
            "FitControls": FitControls,
            "SimulationGrid": SimulationGrid,
            "OutputConfig": OutputConfig,
            "RunConfig": RunConfig,
        }[attribute_name]
    raise AttributeError(f"module 'cmat' has no attribute {attribute_name!r}")

"""
This module provides classes and functions for fitting transit light curves
and simulating transit timing variations (TTVs).
"""
