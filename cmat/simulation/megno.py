"""REBOUND-backed MEGNO simulation helpers."""

from __future__ import annotations

from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import rebound
from tqdm.auto import tqdm

from ..domain.units import ME_TO_MS, MJ_TO_MS
from .execution import build_mass_ratio_parameter_grid, maybe_run_in_pool


def calculate_megno(*, parameters, prop, dt, runtime):
    """Calculate MEGNO for one period-ratio/mass pair."""

    r, mp2 = parameters
    ms = prop[0]["Ms"]
    mp1 = prop[0]["Mp"]
    a1 = prop[0]["orbital_distance"]
    a2 = a1 * r ** (2 / 3)

    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.add(m=ms)
    sim.add(m=mp1 * MJ_TO_MS, a=a1, e=0)
    sim.add(m=mp2 * ME_TO_MS, a=a2, e=0)
    sim.move_to_com()

    period_min = min([sim.particles[1].P, sim.particles[2].P])
    sim.dt = dt * period_min
    sim.init_megno()
    sim.exit_max_distance = 20.0
    try:
        sim.integrate(runtime * period_min, exact_finish_time=0)
        return sim.calculate_megno()
    except rebound.Escape:
        return 10.0


def save_megno_grid_cache(
    *,
    cache_path,
    period_ratios,
    companion_masses,
    megno_results,
):
    """Persist MEGNO grid results as a compressed `.npz` bundle."""

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            period_ratios=np.asarray(period_ratios, dtype=float),
            companion_masses=np.asarray(companion_masses, dtype=float),
            megno_results=np.asarray(megno_results, dtype=float),
        )
    return cache_path


def load_megno_grid_cache(*, cache_path):
    """Load MEGNO grid results from a compressed `.npz` bundle."""

    with np.load(cache_path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def run_megno_grid(
    *,
    period_ratios,
    companion_masses,
    prop,
    dt,
    runtime,
    worker_count=1,
    start_method="fork",
    show_progress=True,
    number_of_threads=None,
    use_cache=False,
    cache_path=None,
    overwrite_cache=False,
    get_context_fn=get_context,
    progress_wrapper=tqdm,
):
    """Run MEGNO over a grid while preserving legacy ordering and caching."""

    if use_cache and cache_path is not None and Path(cache_path).exists():
        if not overwrite_cache:
            payload = load_megno_grid_cache(cache_path=cache_path)
            return payload["megno_results"].tolist()

    parameters = build_mass_ratio_parameter_grid(period_ratios, companion_masses)
    worker = partial(calculate_megno, prop=prop, dt=dt, runtime=runtime)
    results = maybe_run_in_pool(
        worker,
        parameters,
        worker_count=worker_count if number_of_threads is None else number_of_threads,
        start_method=start_method,
        show_progress=show_progress,
        get_context_fn=get_context_fn,
        progress_wrapper=progress_wrapper,
    )
    if use_cache and cache_path is not None:
        save_megno_grid_cache(
            cache_path=cache_path,
            period_ratios=period_ratios,
            companion_masses=companion_masses,
            megno_results=results,
        )
    return results
