"""REBOUND-backed transit-timing simulation helpers."""

from __future__ import annotations

import numpy as np
import rebound

from ..domain.units import ME_TO_MS, MJ_TO_MS

RJ_TO_RS = 0.102792236
LEGACY_RS_TO_AU = 0.00464913034


def calculate_rebound_ttv(
    *,
    parameters,
    prop,
    n_transit_simulations,
    e1=0,
    e2=0,
    inc1=0,
    inc2=0,
    f1=0,
    f2=0,
):
    """Simulate one REBOUND TTV series for a single period-ratio/mass pair."""

    r, mp2 = parameters
    ms = prop[0]["Ms"]
    mp1 = prop[0]["Mp"]
    a1 = prop[0]["orbital_distance"]
    a2 = a1 * r ** (2 / 3)
    rstar = prop[0]["Rs"]
    rp = prop[0]["Rp"]

    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.ri_whfast.safe_mode = 0
    sim.collision = "direct"
    sim.add(m=ms, r=rstar * LEGACY_RS_TO_AU)
    sim.add(
        m=mp1 * MJ_TO_MS,
        r=rp * RJ_TO_RS * LEGACY_RS_TO_AU,
        a=a1,
        e=e1,
        inc=inc1,
        f=f1,
    )
    sim.add(m=mp2 * ME_TO_MS, a=a2, e=e2, inc=inc2, f=f2)
    sim.move_to_com()
    sim.exit_max_distance = 5.0

    period_min = min([sim.particles[1].P, sim.particles[2].P])
    transit_count = int(n_transit_simulations)
    transittimes = np.zeros(transit_count)
    particles = sim.particles
    index = 0
    while index < transit_count:
        y_old = particles[1].y - particles[0].y
        t_old = sim.t
        try:
            sim.integrate(sim.t + period_min / 4)
        except rebound.Escape:
            break
        except rebound.Collision:
            break
        t_new = sim.t
        if y_old * (particles[1].y - particles[0].y) < 0.0 and (
            particles[1].x - particles[0].x > 0.0
        ):
            while t_new - t_old > 1e-9:
                if y_old * (particles[1].y - particles[0].y) < 0.0:
                    t_new = sim.t
                else:
                    t_old = sim.t
                try:
                    sim.integrate((t_new + t_old) / 2.0)
                except (rebound.Escape, rebound.Collision):
                    break
            transittimes[index] = sim.t
            index += 1
            try:
                sim.integrate(sim.t + 5e-5)
            except (rebound.Escape, rebound.Collision):
                break

    if index < transit_count:
        return np.full(transit_count, np.nan)

    c, m = np.linalg.lstsq(
        np.vstack([np.ones(transit_count), range(transit_count)]).T,
        transittimes,
        rcond=None,
    )[0]
    return (transittimes - m * np.array(range(transit_count)) - c) * (
        3600 * 24.0 * 365.0 / 2.0 / np.pi
    )
