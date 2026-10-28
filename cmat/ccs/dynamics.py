"""A deliberately bounded, circular-coplanar REBOUND pilot simulator.

Unlike legacy cmat.simulation.rebound_ttv, this returns RAW conjunction times
and barycentric stellar positions. No best-fit line is removed in this layer.
The star and inner planet have nonzero masses. Observer is along +z; edge-on
transits are relative x=0 crossings with planet in front (+z).

This pilot uses IAS15 + bisection for correctness, not maximum throughput.
Independent scientific cross-checks and integration convergence are REQUIRED
before publication or bulk target searches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ltte import C_AU_PER_DAY, G_AU3_MSUN_DAY2, MJUP_TO_MSUN, SECONDS_PER_DAY


@dataclass(frozen=True)
class TransitSeries:
    epoch: np.ndarray
    transit_day: np.ndarray
    star_z_au: np.ndarray

    @property
    def ltte_sec(self) -> np.ndarray:
        return -self.star_z_au * SECONDS_PER_DAY / C_AU_PER_DAY


def _position(sim) -> tuple[float, float]:
    star, planet = sim.particles[0], sim.particles[1]
    return float(planet.x-star.x), float(planet.z-star.z)


def _refine_crossing(snapshot, left: float, right: float, tol_days: float):
    """Bisect event from an immutable left-bracket state (safe for IAS15)."""
    while right-left > tol_days:
        middle = (left+right)/2
        candidate = snapshot.copy()
        candidate.integrate(middle)
        if _position(candidate)[0] > 0:
            left = middle
        else:
            right = middle
    root = snapshot.copy()
    root.integrate((left+right)/2)
    return root


def simulate_transits(
    epochs: np.ndarray,
    *,
    star_mass_msun: float,
    inner_mass_mjup: float,
    inner_period_day: float,
    outer_mass_mjup: float = 0.0,
    outer_period_day: float | None = None,
    outer_phase_deg: float = 0.0,
    sample_step_fraction: float = 1 / 16,
    max_epoch_span: int = 3000,
) -> TransitSeries:
    """Evaluate requested integer epochs; never initiate a background grid.

    Restricted to circular, edge-on and coplanar systems; requires rebound.
    epoch zero is an unperturbed transit at t=0 before adding the companion.
    No claims about eccentric/inclined or multi-planet systems.
    """
    e = np.asarray(epochs)
    if e.ndim != 1 or e.size == 0 or not np.isfinite(e).all():
        raise ValueError("Finite 1D epochs required")
    if not np.all(e == np.rint(e)):
        raise ValueError("Epochs must be integers")
    e = e.astype(int)
    if len(set(e.tolist())) != len(e):
        raise ValueError("Provide unique epochs; repeat matched model times for duplicate observations")
    emin, emax = int(e.min()), int(e.max())
    if emax-emin > max_epoch_span:
        raise ValueError("Pilot safety cap exceeded: epoch span")
    if min(star_mass_msun, inner_mass_mjup, inner_period_day) <= 0:
        raise ValueError("Star, inner mass and period must be positive")
    if outer_mass_mjup < 0:
        raise ValueError("Outer companion mass must be nonnegative")
    if outer_mass_mjup > 0 and (outer_period_day is None or outer_period_day <= inner_period_day):
        raise ValueError("Outer companion must have a longer positive period")
    if not 0 < sample_step_fraction <= 1 / 8:
        raise ValueError("Require sample_step_fraction in (0, 1/8]")
    try:
        import rebound
    except ImportError as exc:
        raise ImportError("REBOUND is required for numerical CCS pilot simulation") from exc
    sim = rebound.Simulation()
    sim.G = G_AU3_MSUN_DAY2
    sim.integrator = "ias15"
    sim.dt = inner_period_day / 30
    sim.add(m=star_mass_msun)
    # True anomaly f=90 deg means inner planet starts directly in front of star.
    sim.add(m=inner_mass_mjup*MJUP_TO_MSUN, P=inner_period_day,
            inc=np.pi/2, f=np.pi/2, primary=sim.particles[0])
    if outer_mass_mjup > 0:
        # Circular Jacobi outer orbit around barycenter of star+inner planet.
        sim.add(m=outer_mass_mjup*MJUP_TO_MSUN, P=outer_period_day,
                inc=np.pi/2, f=np.deg2rad(outer_phase_deg), primary=sim.calculate_com())
    sim.move_to_com()
    t_start = (emin-0.4) * inner_period_day
    t_end = (emax+0.4) * inner_period_day
    sim.integrate(t_start)
    step = inner_period_day*sample_step_fraction
    # Prevent accidental prohibitively expensive high-precision integrations.
    if (t_end-t_start)/step > 100_000:
        raise ValueError("Pilot step cap exceeded")
    result: dict[int, tuple[float, float]] = {}
    tol_days = 1e-10
    while sim.t < t_end:
        start = sim.copy()
        t_left = float(start.t)
        x_left, _ = _position(start)
        t_next = min(t_end, t_left + step)
        sim.integrate(t_next)
        x_right, z_right = _position(sim)
        if x_left > 0 >= x_right and z_right > 0:
            root = _refine_crossing(start, t_left, t_next, tol_days)
            transit_t = float(root.t)
            epoch = int(np.rint(transit_t/inner_period_day))
            if abs(transit_t/inner_period_day-epoch) > 0.25:
                raise ValueError("Ambiguous epoch match: large dynamical TTV")
            if epoch in result:
                raise ValueError("Duplicate transit epoch in dynamical integration")
            result[epoch] = (transit_t, float(root.particles[0].z))
    missing = sorted(set(e.tolist())-set(result))
    if missing:
        raise ValueError(f"Missing simulated transit epochs: {missing[:6]}")
    return TransitSeries(
        epoch=e,
        transit_day=np.array([result[int(k)][0] for k in e]),
        star_z_au=np.array([result[int(k)][1] for k in e]),
    )


def companion_signals(
    epochs: np.ndarray,
    *,
    star_mass_msun: float,
    inner_mass_mjup: float,
    inner_period_day: float,
    outer_mass_mjup: float,
    outer_period_day: float,
    outer_phase_deg: float,
    max_epoch_span: int = 3000,
) -> dict[str, np.ndarray]:
    """Subtract no-outer reference, returning dynamical / LTTE in seconds."""
    kwargs = dict(
        star_mass_msun=star_mass_msun,
        inner_mass_mjup=inner_mass_mjup,
        inner_period_day=inner_period_day,
        max_epoch_span=max_epoch_span,
    )
    baseline = simulate_transits(epochs, **kwargs)
    companion = simulate_transits(
        epochs, **kwargs,
        outer_mass_mjup=outer_mass_mjup,
        outer_period_day=outer_period_day,
        outer_phase_deg=outer_phase_deg,
    )
    dyn = (companion.transit_day-baseline.transit_day)*SECONDS_PER_DAY
    ltte = companion.ltte_sec-baseline.ltte_sec
    return {
        "M0": np.zeros_like(dyn),
        "M1": dyn,
        "M2": ltte,
        "M3": dyn+ltte,
    }
