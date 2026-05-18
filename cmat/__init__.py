name = "fitransit"

__all__ = [
    "Fitlpf",
    "TransitFitWorkflow",
    "ttv_sim",
    "TTVSimulation",
    "TargetConfig",
    "FitControls",
    "SimulationGrid",
    "ExecutionConfig",
    "BayesianScoringConfig",
    "ScoringConfig",
    "OutputConfig",
    "RunConfig",
]


def __getattr__(attribute_name):
    if attribute_name in {"Fitlpf", "TransitFitWorkflow"}:
        from .base import Fitlpf, TransitFitWorkflow

        return {
            "Fitlpf": Fitlpf,
            "TransitFitWorkflow": TransitFitWorkflow,
        }[attribute_name]
    if attribute_name in {"ttv_sim", "TTVSimulation"}:
        from .ttv_sim import TTVSimulation, ttv_sim

        return {
            "ttv_sim": ttv_sim,
            "TTVSimulation": TTVSimulation,
        }[attribute_name]
    if attribute_name in {
        "TargetConfig",
        "FitControls",
        "SimulationGrid",
        "ExecutionConfig",
        "BayesianScoringConfig",
        "ScoringConfig",
        "OutputConfig",
        "RunConfig",
    }:
        from .config import (
            BayesianScoringConfig,
            ExecutionConfig,
            FitControls,
            OutputConfig,
            RunConfig,
            ScoringConfig,
            SimulationGrid,
            TargetConfig,
        )

        return {
            "TargetConfig": TargetConfig,
            "FitControls": FitControls,
            "SimulationGrid": SimulationGrid,
            "ExecutionConfig": ExecutionConfig,
            "BayesianScoringConfig": BayesianScoringConfig,
            "ScoringConfig": ScoringConfig,
            "OutputConfig": OutputConfig,
            "RunConfig": RunConfig,
        }[attribute_name]
    raise AttributeError(f"module 'cmat' has no attribute {attribute_name!r}")


"""
This module provides classes and functions for fitting transit light curves
and simulating transit timing variations (TTVs).
"""
