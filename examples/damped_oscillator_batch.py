"""Batch-style deployment example for the Stage 5 simulator-adapter workflow.

This example keeps the same damped-oscillator forward model as the smaller
simulator-adapter demo, but stores the run under a deterministic artifact
layout so the same command can be re-run and compared later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cmat.config import ExecutionConfig
from cmat.workflow import (
    run_simulator_adapter,
    write_simulator_adapter_manifest,
    write_simulator_adapter_portfolio,
)

from damped_oscillator_adapter import DampedOscillatorAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the damped-oscillator adapter in a batch-style artifact layout.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("artifacts") / "deployment",
        help="Root directory that will contain deployment-style run folders.",
    )
    parser.add_argument(
        "--run-name",
        default="damped-oscillator-baseline",
        help="Stable run folder name for reproducible reruns.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="ExecutionConfig.worker_count passed to the adapter workflow.",
    )
    parser.add_argument(
        "--start-method",
        default="fork",
        help="ExecutionConfig.start_method passed to the adapter workflow.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display tqdm progress for larger parameter grids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execution = ExecutionConfig(
        worker_count=args.worker_count,
        start_method=args.start_method,
        show_progress=args.show_progress,
    )
    run_dir = args.root_dir / args.run_name
    portfolio_dir = run_dir / "portfolio"
    metadata_path = run_dir / "run_metadata.json"

    result = run_simulator_adapter(
        DampedOscillatorAdapter(),
        execution=execution,
    )
    portfolio_paths = write_simulator_adapter_portfolio(
        result,
        output_dir=portfolio_dir,
        title="Damped oscillator deployment example",
    )
    manifest_path = write_simulator_adapter_manifest(
        result,
        adapter_name="damped_oscillator",
        metadata_path=metadata_path,
        execution=execution,
        notes={
            "artifact_layout": "root_dir/run_name/{run_metadata.json, portfolio/...}",
            "rerun_strategy": "reuse the same --run-name to overwrite a deterministic run folder",
        },
    )

    print("Damped oscillator deployment example")
    print("summary:", result.summary)
    print("run directory:", run_dir)
    print("run metadata:", manifest_path)
    print(
        "portfolio outputs:",
        {name: str(path) for name, path in sorted(portfolio_paths.items())},
    )


if __name__ == "__main__":
    main()
