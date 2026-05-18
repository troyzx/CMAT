import os
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy.stats
from tqdm.auto import tqdm

from . import cache
from . import scoring as _scoring
from .scoring import (
    BAYESIAN_MASS_THRESHOLD_BACKEND,
    Chi2AndRmsMassThresholdScorer,
    MassThresholds,
)
from .simulation.execution import (
    build_mass_ratio_parameter_grid,
    resolve_worker_count,
)

mj_to_ms = 9.5e-4
me_to_ms = 3.0e-6
rj_to_rs = 0.102792236
rs_to_AU = 0.00464913034
daytos = 24 * 60 * 60

get_chi2 = _scoring.get_chi2
get_rms = _scoring.get_rms


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
    def __init__(
        self, epochs, ttv_mcmc, ttv_err, rs, mp2s, prop, N=80, scoring_backend=None
    ):
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
        return resolve_worker_count(number_of_threads, worker_count=self.worker_count)

    def _progress_iterator(self, iterator, *, total):
        if self.show_progress:
            return tqdm(iterator, total=total)
        return iterator

    def _legacy_transit_simulation_count(self):
        return int((self.epochs[-1] - self.epochs[0]) * 2)

    def calculate_rebound(self, par, e1=0, e2=0, inc1=0, inc2=0, f1=0, f2=0):
        from .simulation.rebound_ttv import calculate_rebound_ttv

        return calculate_rebound_ttv(
            parameters=par,
            prop=self.prop,
            n_transit_simulations=self._legacy_transit_simulation_count(),
            e1=e1,
            e2=e2,
            inc1=inc1,
            inc2=inc2,
            f1=f1,
            f2=f2,
        )

    def get_ttv_rebound_all(
        self,
        number_of_thread=None,
        *,
        use_cache=False,
        cache_path=None,
        overwrite_cache=False,
    ):
        if use_cache and cache_path is None:
            raise ValueError(
                "cache_path is required when use_cache=True for get_ttv_rebound_all()"
            )
        if use_cache and cache_path is not None and not overwrite_cache:
            if os.path.exists(cache_path):
                self.load_ttv_grid_cache(cache_path)
                return self.ttv_rebound

        parameters = build_mass_ratio_parameter_grid(self.rs, self.mp2s)
        with get_context(self.start_method).Pool(
            self._resolve_worker_count(number_of_thread)
        ) as pool:
            self.ttv_results = list(
                self._progress_iterator(
                    pool.imap(self.calculate_rebound, parameters),
                    total=len(parameters),
                )
            )
        self.ttv_rebound = np.array(self.ttv_results)
        if use_cache and cache_path is not None:
            self.save_ttv_grid_cache(cache_path)
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
            raise ValueError(
                "chi2_surface is only available from the chi2_rms scoring backend"
            )
        return np.asarray(thresholds.chi2_surface, dtype=float)

    def get_relative_log_likelihood_surface(self):
        """Return the latest relative Gaussian log-likelihood proxy, -0.5 * chi2."""

        thresholds = self.get_mass_thresholds()
        if thresholds.relative_log_likelihood_surface is None:
            raise ValueError(
                "relative_log_likelihood_surface is only available from the chi2_rms scoring backend"
            )
        return np.asarray(thresholds.relative_log_likelihood_surface, dtype=float)

    def get_reduced_chi2_surface(self, *, degrees_of_freedom=None):
        """Return chi2 / dof on the current (companion_mass, period_ratio) grid."""

        thresholds = self.get_mass_thresholds()
        if thresholds.chi2_surface is None:
            raise ValueError(
                "reduced_chi2_surface is only available from the chi2_rms scoring backend"
            )
        if degrees_of_freedom is None:
            if thresholds.reduced_chi2_surface is not None:
                return np.asarray(thresholds.reduced_chi2_surface, dtype=float)
            degrees_of_freedom = thresholds.chi2_degrees_of_freedom
        if degrees_of_freedom is None:
            raise ValueError(
                "degrees_of_freedom is required when no stored default is available"
            )
        if degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be positive")
        return np.asarray(thresholds.chi2_surface, dtype=float) / float(
            degrees_of_freedom
        )

    def plot_chi2_contour(
        self,
        *,
        statistic="chi2",
        degrees_of_freedom=None,
        levels=None,
        vmin=None,
        vmax=None,
        ax=None,
        cmap="plasma",
        show_threshold=True,
        threshold_color="white",
        figsize=(4, 3.2),
        dpi=200,
    ):
        """Plot a contour map of chi2 or relative Gaussian log-likelihood proxy.

        The score surface is shaped as (len(companion_masses), len(period_ratios)),
        with rows corresponding to companion masses and columns corresponding to
        period ratios.
        """

        from .plotting.score_surfaces import plot_score_surface

        thresholds = self.get_mass_thresholds()
        if statistic == "chi2":
            surface = self.get_chi2_surface()
            colorbar_label = r"$\chi^2$"
            threshold = thresholds.chi2_threshold
        elif statistic == "reduced_chi2":
            surface = self.get_reduced_chi2_surface(
                degrees_of_freedom=degrees_of_freedom
            )
            colorbar_label = r"reduced $\chi^2$"
            if degrees_of_freedom is None:
                degrees_of_freedom = thresholds.chi2_degrees_of_freedom
            threshold = (
                None
                if thresholds.chi2_threshold is None or degrees_of_freedom is None
                else thresholds.chi2_threshold / float(degrees_of_freedom)
            )
        elif statistic in {"relative_log_likelihood", "log_likelihood", "loglike"}:
            surface = self.get_relative_log_likelihood_surface()
            colorbar_label = r"relative log likelihood proxy $(-\chi^2 / 2)$"
            threshold = None
        else:
            raise ValueError(
                "statistic must be 'chi2', 'reduced_chi2', or 'relative_log_likelihood'"
            )
        if statistic in {"chi2", "reduced_chi2"}:
            if vmin is None:
                vmin = 0.0
            if vmax is None and threshold is not None:
                vmax = threshold

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
        return plot_score_surface(
            period_ratios=period_ratios,
            companion_masses=companion_masses,
            surface=surface,
            statistic_label=colorbar_label,
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            show_threshold=show_threshold and statistic != "relative_log_likelihood",
            threshold_color=threshold_color,
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
        )

    def simulation_m(self, par):
        from .simulation.megno import calculate_megno

        return calculate_megno(
            parameters=par,
            prop=self.prop,
            dt=self.megno_dt,
            runtime=self.megno_runtime,
        )

    # Run the MEGNO simulations for all parameter combinations
    def run_megno(
        self,
        number_of_threads=None,
        *,
        use_cache=False,
        cache_path=None,
        overwrite_cache=False,
    ):
        if use_cache and cache_path is None:
            raise ValueError("cache_path is required when use_cache=True for run_megno()")
        if use_cache and cache_path is not None and Path(cache_path).exists():
            if not overwrite_cache:
                self.load_megno_grid_cache(cache_path)
                return self.megno_results

        parameters = build_mass_ratio_parameter_grid(self.rs, self.mp2s)
        with get_context(self.start_method).Pool(
            self._resolve_worker_count(number_of_threads)
        ) as pool:
            self.megno_results = list(
                self._progress_iterator(
                    pool.imap(self.simulation_m, parameters),
                    total=len(parameters),
                )
            )
        if use_cache and cache_path is not None:
            self.save_megno_grid_cache(cache_path)
        return self.megno_results

    # Plot the MEGNO results as a 2D color map
    def plot_megno(self):
        from .plotting.megno import plot_megno_surface

        return plot_megno_surface(
            period_ratios=self.rs,
            companion_masses=self.mp2s,
            megno_results=self.megno_results,
        )

    def save_ttv_grid_cache(self, cache_path):
        if not self._has_nonempty_results("ttv_results"):
            raise ValueError("No TTV results to save. Run get_ttv_rebound_all() first.")
        cache.save_ttv_grid(
            cache_path,
            period_ratios=self.rs,
            companion_masses=self.mp2s,
            epochs=self.epochs,
            ttv_mcmc=self.ttv_mcmc,
            ttv_err=self.ttv_err,
            ttv_results=self.ttv_results,
        )
        return Path(cache_path)

    def load_ttv_grid_cache(self, cache_path):
        cached = cache.load_ttv_grid(cache_path)
        cache.validate_ttv_grid_compatibility(
            cached,
            period_ratios=self.rs,
            companion_masses=self.mp2s,
            epochs=self.epochs,
            ttv_mcmc=self.ttv_mcmc,
            ttv_err=self.ttv_err,
        )
        self.ttv_results = list(cached["ttv_results"])
        self.ttv_rebound = np.array(self.ttv_results)
        return cached

    def save_megno_grid_cache(self, cache_path):
        if not self._has_nonempty_results("megno_results"):
            raise ValueError("No MEGNO results to save. Run run_megno() first.")
        cache.save_megno_grid(
            cache_path,
            period_ratios=self.rs,
            companion_masses=self.mp2s,
            megno_results=self.megno_results,
        )
        return Path(cache_path)

    def load_megno_grid_cache(self, cache_path):
        cached = cache.load_megno_grid(cache_path)
        cache.validate_megno_grid_compatibility(
            cached, period_ratios=self.rs, companion_masses=self.mp2s
        )
        self.megno_results = list(cached["megno_results"])
        return cached

    def save_scoring_summary(self, path):
        if not hasattr(self, "mass_thresholds"):
            raise ValueError("No scoring summary to save. Run get_m_crit() first.")
        cache.save_scoring_summary(path, mass_thresholds=self.mass_thresholds)
        return Path(path)

    def load_scoring_summary(self, path):
        data = cache.load_scoring_summary(path)
        self.mass_thresholds = MassThresholds.from_dict(data)
        self.m_crit_chi2 = self.mass_thresholds.chi2
        self.m_crit_rms = self.mass_thresholds.rms
        return data

    def save_checkpoint(self, run_dir):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        if self._has_nonempty_results("ttv_results"):
            self.save_ttv_grid_cache(run_dir / "ttv_grid.npz")

        if self._has_nonempty_results("megno_results"):
            self.save_megno_grid_cache(run_dir / "megno_grid.npz")

        if hasattr(self, "mass_thresholds"):
            self.save_scoring_summary(run_dir / "scoring_summary.npz")
        return run_dir

    def load_checkpoint(self, run_dir):
        run_dir = Path(run_dir)

        ttv_path = run_dir / "ttv_grid.npz"
        if ttv_path.exists():
            self.load_ttv_grid_cache(ttv_path)

        megno_path = run_dir / "megno_grid.npz"
        if megno_path.exists():
            self.load_megno_grid_cache(megno_path)

        scoring_path = run_dir / "scoring_summary.npz"
        if scoring_path.exists():
            self.load_scoring_summary(scoring_path)
        return run_dir

    def _has_nonempty_results(self, attribute_name):
        if not hasattr(self, attribute_name):
            return False
        results = getattr(self, attribute_name)
        if results is None:
            return False
        try:
            return len(results) > 0
        except TypeError:
            return False


TTVSimulation = ttv_sim
