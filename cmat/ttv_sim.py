import warnings
from numbers import Integral

import rebound
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import get_context
from tqdm.auto import tqdm

from .scoring import (
    BAYESIAN_MASS_THRESHOLD_BACKEND,
    Chi2AndRmsMassThresholdScorer,
    get_chi2,
    get_rms,
)

mj_to_ms = 9.5e-4
me_to_ms = 3.0e-6
rj_to_rs = 0.102792236
rs_to_AU = 0.00464913034
daytos = 24 * 60 * 60


"""
This code defines a class, ttv_sim, which is built to perform simulations of
exoplanetary systems using the REBOUND N-body integrator. The class has
several methods for running simulations, calculating transit times, and
generating plots of the results.

The main functionality of the class is to analyze the stability of planetary
systems by evaluating the Mean Exponential Growth of Nearby Orbits (MEGNO)
values. This indicator can help determine the chaotic behavior of a system
and find potentially stable configurations.
"""


class ttv_sim:
    def __init__(self, epochs, ttv_mcmc, ttv_err, rs, mp2s, prop, N=80, scoring_backend=None):
        self.epochs = epochs  # The epochs at which to calculate the TTVs
        self.ttv_mcmc = ttv_mcmc  # The TTVs obtained from MCMC fitting
        self.ttv_err = ttv_err  # The errors on the MCMC TTVs
        self.rs = rs  # The radii of the planets
        self.mp2s = mp2s  # The masses of the planets
        self.prop = prop  # The orbital properties of the planets
        self.N = N  # The number of TTV simulations to perform
        self.crit = scipy.stats.chi2.ppf(0.997, len(ttv_mcmc))
        self.prop = prop
        self.a1 = prop[0]["orbital_distance"]
        self.t1 = prop[0]["orbital_period"]
        self.mp1 = prop[0]["Mp"]
        self.ms = prop[0]["Ms"]
        self.megno_dt = 1 / 20
        self.megno_runtime = 1e4
        self.worker_count = 1
        self.start_method = "fork"
        self.show_progress = True
        self.ttv_rebound = []
        self.scoring_backend = scoring_backend or Chi2AndRmsMassThresholdScorer()

    def _resolve_worker_count(self, number_of_threads):
        if number_of_threads is None:
            number_of_threads = self.worker_count
        if isinstance(number_of_threads, bool) or not isinstance(number_of_threads, Integral):
            raise TypeError("number_of_threads must be an integer")
        number_of_threads = int(number_of_threads)
        if number_of_threads <= 0:
            raise ValueError("number_of_threads must be positive")
        return number_of_threads

    def _progress_iterator(self, iterator, *, total):
        if self.show_progress:
            return tqdm(iterator, total=total)
        return iterator

    def calculate_rebound(self, par, e1=0, e2=0, inc1=0, inc2=0, f1=0, f2=0):
        r, mp2 = par
        sim = rebound.Simulation()
        ms = self.prop[0]["Ms"]
        mp1 = self.prop[0]["Mp"]
        a1 = self.prop[0]["orbital_distance"]
        a2 = a1 * r ** (2 / 3)
        rstar = self.prop[0]["Rs"]
        rp = self.prop[0]["Rp"]

        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0

        # Collision
        sim.collision = "direct"
        sim.add(m=ms, r=rstar * rs_to_AU)  # Star
        sim.add(
            m=mp1 * mj_to_ms,
            r=rp * rj_to_rs * rs_to_AU,
            a=a1,
            e=e1,
            inc=inc1,
            f=f1,
        )  # Primary planet
        sim.add(
            m=mp2 * me_to_ms,
            a=a2,
            e=e2,
            inc=inc2,
            f=f2
            )  # Companion planet
        sim.move_to_com()
        sim.exit_max_distance = 5.0

        period_min = min([sim.particles[1].P, sim.particles[2].P])
        N = (self.epochs[-1] - self.epochs[0]) * 2
        transittimes = np.zeros(N)
        p = sim.particles
        i = 0
        while i < N:
            y_old = p[1].y - p[0].y
            t_old = sim.t
            try:
                sim.integrate(sim.t + period_min / 4)
                # check for transits every <period_min / 4>
                # Note that <period_min / 4>
                # is shorter than one inner planet's orbit
            except rebound.Escape:
                # print("Escape at r={}, mp2={} when i={}".format(r,mp2,i))
                break
            except rebound.Collision:
                # print("Collide at r={}, mp2={} when i={}".format(r,mp2,i))
                break
            t_new = sim.t
            if y_old * (p[1].y - p[0].y) < 0.0 and p[1].x - p[0].x > 0.0:
                # sign changed (y_old*y<0), planet in front of star (x>0)
                while t_new - t_old > 1e-9:
                    # bisect until prec of 1e-9 reached
                    if y_old * (p[1].y - p[0].y) < 0.0:
                        t_new = sim.t
                    else:
                        t_old = sim.t
                    try:
                        sim.integrate((t_new + t_old) / 2.0)
                    except (rebound.Escape, rebound.Collision):
                        break
                transittimes[i] = sim.t
                i += 1
                try:
                    sim.integrate(sim.t + 5e-5)
                except (rebound.Escape, rebound.Collision):
                    break
        if i < N:
            # Early termination means the transit series is not physically usable
            # as a scoring input. Return an explicit invalid series instead of a
            # zero-filled artifact so downstream scorers can exclude it cleanly.
            return np.full(N, np.nan)
        c, m = np.linalg.lstsq(
            np.vstack([np.ones(N), range(N)]).T, transittimes, rcond=None
        )[0]
        ttv_rebound = (transittimes - m * np.array(range(N)) - c) * (
            3600 * 24.0 * 365.0 / 2.0 / np.pi
        )
        return ttv_rebound

    def get_ttv_rebound_all(self, number_of_thread=None):
        parameters = []
        for mp2 in self.mp2s:
            for r in self.rs:
                parameters.append((r, mp2))
        with get_context(self.start_method).Pool(
            self._resolve_worker_count(number_of_thread)
        ) as p:
            self.ttv_results = list(
                self._progress_iterator(
                    p.imap(self.calculate_rebound, parameters),
                    total=len(parameters),
                )
            )
        self.ttv_rebound = np.array(self.ttv_results)
        return self.ttv_rebound

    def get_m_crit(self):
        thresholds = self.scoring_backend.critical_masses(
            ttv_results=self.ttv_results,
            epoch=self.epochs,
            ttv_mcmc=self.ttv_mcmc,
            ttv_err=self.ttv_err,
            period_ratios=self.rs,
            companion_masses=self.mp2s,
        )
        self.mass_thresholds = thresholds
        self.m_crit_chi2 = thresholds.chi2
        self.m_crit_rms = thresholds.rms
        if thresholds.backend == BAYESIAN_MASS_THRESHOLD_BACKEND:
            warnings.warn(
                "Bayesian scoring is an experimental Stage 4 backend. "
                "Use get_mass_thresholds() for the full posterior mass summary; "
                "get_m_crit() only returns legacy chi2/RMS arrays for backward compatibility.",
                UserWarning,
                stacklevel=2,
            )

        return self.m_crit_chi2, self.m_crit_rms

    def get_critical_masses(self):
        """Return the first rejected companion masses for the current grid."""

        return self.get_m_crit()

    def get_mass_thresholds(self):
        """Return the full MassThresholds object from the latest scoring run."""

        if not hasattr(self, "mass_thresholds"):
            raise ValueError("Run get_m_crit() before requesting mass thresholds")
        return self.mass_thresholds

    def get_scoring_summary(self):
        """Return a JSON-serializable summary of the latest scoring result."""

        return self.get_mass_thresholds().to_dict()

    def get_chi2_surface(self):
        """Return the latest chi-squared surface with shape (len(mp2s), len(rs))."""

        thresholds = self.get_mass_thresholds()
        if thresholds.chi2_surface is None:
            raise ValueError("chi2_surface is only available from the chi2_rms scoring backend")
        return np.asarray(thresholds.chi2_surface, dtype=float)

    def get_relative_log_likelihood_surface(self):
        """Return the latest relative Gaussian log-likelihood proxy, -0.5 * chi2."""

        thresholds = self.get_mass_thresholds()
        if thresholds.relative_log_likelihood_surface is None:
            raise ValueError(
                "relative_log_likelihood_surface is only available from the chi2_rms scoring backend"
            )
        return np.asarray(thresholds.relative_log_likelihood_surface, dtype=float)

    def plot_chi2_contour(
        self,
        *,
        statistic="chi2",
        levels=20,
        ax=None,
        cmap="viridis",
        show_threshold=True,
        threshold_color="white",
    ):
        """Plot a contour map of chi2 or relative Gaussian log-likelihood proxy.

        The score surface is shaped as (len(companion_masses), len(period_ratios)),
        with rows corresponding to companion masses and columns corresponding to
        period ratios.
        """

        thresholds = self.get_mass_thresholds()
        if statistic == "chi2":
            surface = self.get_chi2_surface()
            colorbar_label = r"$\chi^2$"
        elif statistic in {"relative_log_likelihood", "log_likelihood", "loglike"}:
            surface = self.get_relative_log_likelihood_surface()
            colorbar_label = r"relative log likelihood proxy $(-\chi^2 / 2)$"
        else:
            raise ValueError(
                "statistic must be 'chi2' or 'relative_log_likelihood'"
            )

        period_ratios = (
            np.asarray(thresholds.period_ratios, dtype=float)
            if thresholds.period_ratios is not None
            else np.asarray(self.rs, dtype=float)
        )
        companion_masses = (
            np.asarray(thresholds.companion_masses, dtype=float)
            if thresholds.companion_masses is not None
            else np.asarray(self.mp2s, dtype=float)
        )
        if surface.shape != (len(companion_masses), len(period_ratios)):
            raise ValueError("score surface shape must match the configured mp2-r grid")
        if len(period_ratios) < 2 or len(companion_masses) < 2:
            raise ValueError("contour plotting requires at least a 2x2 mp2-r grid")
        if not np.any(np.isfinite(surface)):
            raise ValueError("score surface must contain at least one finite value")
        finite_surface = surface[np.isfinite(surface)]
        if np.allclose(finite_surface, finite_surface[0]):
            raise ValueError("score surface must vary across the grid for contour plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        masked_surface = np.ma.masked_invalid(surface)
        contour = ax.contourf(
            period_ratios,
            companion_masses,
            masked_surface,
            levels=levels,
            cmap=cmap,
        )
        colorbar = fig.colorbar(contour, ax=ax)
        colorbar.set_label(colorbar_label)

        if (
            statistic == "chi2"
            and show_threshold
            and thresholds.chi2_threshold is not None
            and np.isfinite(thresholds.chi2_threshold)
            and np.nanmin(surface) <= thresholds.chi2_threshold <= np.nanmax(surface)
        ):
            threshold_contour = ax.contour(
                period_ratios,
                companion_masses,
                masked_surface,
                levels=[thresholds.chi2_threshold],
                colors=threshold_color,
                linewidths=1.2,
            )
            ax.clabel(threshold_contour, fmt={thresholds.chi2_threshold: "chi2 limit"})

        ax.set_xlabel(r"$P_2/P_1$")
        ax.set_ylabel(r"$M_2$ [$M_\oplus$]")
        ax.set_title("TTV score surface")
        ax.grid(alpha=0.25)
        return fig, ax

    def simulation_m(self, par):
        r, mp2 = par  # unpack parameters
        prop = self.prop
        ms = prop[0]["Ms"]
        mp1 = prop[0]["Mp"]
        a1 = prop[0]["orbital_distance"]
        a2 = a1 * r ** (2 / 3)

        sim = rebound.Simulation()
        sim.integrator = "whfast"
        sim.ri_whfast.safe_mode = 0
        sim.add(m=ms)  # Star
        sim.add(m=mp1 * mj_to_ms, a=a1, e=0)  # Primary planet
        sim.add(m=mp2 * me_to_ms, a=a2, e=0)  # Companion planet
        sim.move_to_com()

        period_min = min([sim.particles[1].P, sim.particles[2].P])
        sim.dt = self.megno_dt * period_min

        sim.init_megno()
        sim.exit_max_distance = 20.0
        try:
            sim.integrate(self.megno_runtime * period_min, exact_finish_time=0)
            # integrate for <runtime>, integrating to the nearest
            # timestep for each output to keep the timestep constant
            # and preserve WHFast's symplectic nature
            megno = sim.calculate_megno()
            return megno
        except rebound.Escape:
            return 10.0
        # At least one particle got ejected,
        # returning large MEGNO.

    # Run the MEGNO simulations for all parameter combinations
    def run_megno(self, number_of_threads=None):
        rs = self.rs
        mp2s = self.mp2s
        parameters = []
        for mp2 in mp2s:
            for r in rs:
                parameters.append((r, mp2))

        with get_context(self.start_method).Pool(
            self._resolve_worker_count(number_of_threads)
        ) as p:
            self.megno_results = list(
                self._progress_iterator(
                    p.imap(self.simulation_m, parameters),
                    total=len(parameters),
                )
            )
        return self.megno_results

    # Plot the MEGNO results as a 2D color map
    def plot_megno(self):
        rs = self.rs
        mp2s = self.mp2s
        results2d = np.array(self.megno_results).reshape(len(rs), len(mp2s))
        fig, ax = plt.subplots(figsize=(7, 5))
        extent = [min(rs), max(rs), min(mp2s), max(mp2s)]
        ax.set_xlim(extent[0], extent[1])
        ax.set_xlabel("$P_2/P_1$")
        ax.set_ylim(extent[2], extent[3])
        ax.set_ylabel("Mass [$M_j$]")
        im = ax.imshow(
            results2d,
            interpolation="none",
            vmin=1.9,
            vmax=10,
            cmap="RdYlGn_r",
            origin="lower",
            aspect="auto",
            extent=extent,
            alpha=0.8,
        )
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("MEGNO $\\langle Y \\rangle$")
        plt.grid()
        new_ticks = [1.5, 2, 2.5, 3, 3.3, 3.5, 3.8, 4]
        plt.xticks(new_ticks)
        plt.xlabel(r"$P_2/P_1$")
        plt.ylabel(r"$M_2$ [$M_j$]")


TTVSimulation = ttv_sim
