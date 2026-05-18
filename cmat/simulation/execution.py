"""Import-safe execution helpers for simulation grids."""

from __future__ import annotations

from multiprocessing import get_context
from numbers import Integral

from tqdm.auto import tqdm


def build_mass_ratio_parameter_grid(period_ratios, companion_masses):
    """Return legacy-ordered `(period_ratio, companion_mass)` pairs."""

    return [
        (float(period_ratio), float(companion_mass))
        for companion_mass in companion_masses
        for period_ratio in period_ratios
    ]


def resolve_worker_count(number_of_threads=None, *, worker_count=1):
    """Resolve the effective worker count from legacy and configured inputs."""

    if number_of_threads is None:
        number_of_threads = worker_count
    if isinstance(number_of_threads, bool) or not isinstance(
        number_of_threads, Integral
    ):
        raise TypeError("number_of_threads must be an integer")
    number_of_threads = int(number_of_threads)
    if number_of_threads <= 0:
        raise ValueError("number_of_threads must be positive")
    return number_of_threads


def maybe_run_in_pool(
    worker,
    parameters,
    *,
    worker_count,
    start_method="fork",
    show_progress=True,
    get_context_fn=get_context,
    progress_wrapper=tqdm,
):
    """Run a picklable worker over a parameter grid, serially or in a pool."""

    parameters = list(parameters)
    if worker_count == 1:
        iterator = map(worker, parameters)
        if show_progress:
            iterator = progress_wrapper(iterator, total=len(parameters))
        return list(iterator)

    with get_context_fn(start_method).Pool(worker_count) as pool:
        iterator = pool.imap(worker, parameters)
        if show_progress:
            iterator = progress_wrapper(iterator, total=len(parameters))
        return list(iterator)
