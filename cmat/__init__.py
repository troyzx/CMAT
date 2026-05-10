name = "fitransit"

__all__ = ["Fitlpf", "ttv_sim"]


def __getattr__(name):
    if name == "Fitlpf":
        from .base import Fitlpf

        return Fitlpf
    if name == "ttv_sim":
        from .ttv_sim import ttv_sim

        return ttv_sim
    raise AttributeError(f"module 'cmat' has no attribute {name!r}")

"""
This module provides classes and functions for fitting transit light curves
and simulating transit timing variations (TTVs).
"""
