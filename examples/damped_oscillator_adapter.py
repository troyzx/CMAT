"""Demonstrate the Stage 5 simulator-adapter interface on a non-astronomy system.

This example uses a damped oscillator as a tiny stand-in for a generic
physics-based forward model. The adapter owns the observed signal, latent-state
grid, scoring rule, and summary logic; `cmat.workflow.run_simulator_adapter(...)`
runs that workflow shape without touching the astronomy-specific TTV stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cmat.config import ExecutionConfig
from cmat.workflow import run_simulator_adapter, write_simulator_adapter_portfolio


class DampedOscillatorAdapter:
    """Small deterministic adapter for a second-order damped oscillator."""

    def __init__(self) -> None:
        self.sample_times = np.linspace(0.0, 4.0, 25)
        self.observed_displacement = self._displacement(stiffness=4.0, damping=0.6)
        self.acceptance_threshold = 0.08

    def parameter_grid(self):
        return [
            {"stiffness": stiffness, "damping": damping}
            for stiffness in (3.0, 4.0, 5.0)
            for damping in (0.3, 0.6, 0.9)
        ]

    def _displacement(self, *, stiffness: float, damping: float) -> np.ndarray:
        angular_frequency = np.sqrt(max(stiffness - 0.25 * damping**2, 0.05))
        return np.exp(-0.5 * damping * self.sample_times) * np.cos(
            angular_frequency * self.sample_times
        )

    def simulate(self, parameters):
        return self._displacement(
            stiffness=parameters["stiffness"],
            damping=parameters["damping"],
        )

    def score(self, simulated_observable) -> float:
        residual = np.asarray(simulated_observable, dtype=float) - self.observed_displacement
        return float(np.sqrt(np.mean(residual**2)))

    def is_accepted(self, score: float) -> bool:
        return score <= self.acceptance_threshold

    def summarize(self, *, parameter_grid, scores, accepted):
        best_index = int(np.argmin(scores))
        return {
            "acceptance_threshold": self.acceptance_threshold,
            "accepted_count": int(np.sum(accepted)),
            "best_parameters": parameter_grid[best_index],
            "best_score": float(scores[best_index]),
        }


def main() -> None:
    adapter = DampedOscillatorAdapter()
    result = run_simulator_adapter(
        adapter,
        execution=ExecutionConfig(worker_count=1, show_progress=False),
    )
    portfolio_paths = write_simulator_adapter_portfolio(
        result,
        output_dir=Path("artifacts") / "damped-oscillator-portfolio",
        title="Damped oscillator simulator-adapter demo",
    )

    print("Damped oscillator adapter example")
    print("summary:", result.summary)
    print("accepted:", result.accepted.tolist())
    print(
        "portfolio outputs:",
        {name: str(path) for name, path in sorted(portfolio_paths.items())},
    )


if __name__ == "__main__":
    main()
