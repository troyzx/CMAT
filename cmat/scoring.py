"""Scoring helpers and backend interfaces for comparing simulated TTV grids."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
from typing import Protocol

import emcee
import numpy as np
import scipy.stats
from scipy.special import logsumexp

from .config import SUPPORTED_BAYESIAN_NUISANCE_PARAMETERS

DEFAULT_MASS_THRESHOLD_BACKEND = "chi2_rms"
BAYESIAN_MASS_THRESHOLD_BACKEND = "bayesian"
INVALID_MODEL_LOG_SCORE = -1e300


def get_chi2(ttv_rebound, epoch, ttv_mcmc, ttv_err):
    """Return the best chi-squared score over possible epoch alignments."""

    rangea = range(epoch[-1] - epoch[0])
    T = [ttv_rebound[np.array(epoch - epoch[0]) + a] for a in rangea]
    chi2 = (((T - ttv_mcmc) ** 2) / ttv_err**2).sum(axis=1)
    return chi2.min()


def get_rms(ttv_rebound):
    """Return the root-mean-square amplitude of simulated TTV residuals."""

    rms = np.sqrt(np.mean(ttv_rebound**2))
    return rms


@dataclass(frozen=True)
class BayesianPosteriorInterval:
    """Posterior interval summary for one nuisance parameter."""

    median: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class BayesianMassPosterior:
    """Experimental Bayesian mass summary for one period ratio.

    `upper_bound` is a backward-compatible alias for `credible_upper_bound`.
    New code should prefer `credible_upper_bound` or `rejection_upper_bound`
    explicitly.
    """

    period_ratio: float
    masses: np.ndarray
    log_evidence: np.ndarray
    model_probabilities: np.ndarray
    cumulative_probability: np.ndarray
    posterior_predictive_score: np.ndarray
    best_mass: float | None
    credible_upper_bound: float | None
    rejection_upper_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True)
class BayesianMassLimitCurve:
    """Experimental Bayesian mass-summary surface keyed by period ratio.

    `upper_bound` is a backward-compatible alias for `credible_upper_bound`.
    New code should prefer `credible_upper_bound` or `rejection_upper_bound`
    explicitly.
    """

    period_ratios: np.ndarray
    evaluated_masses: np.ndarray
    credible_upper_bound: tuple[float | None, ...]
    rejection_upper_bound: tuple[float | None, ...]
    upper_bound: tuple[float | None, ...]
    units: str = "earth_masses"
    posterior_by_period_ratio: tuple[BayesianMassPosterior, ...] = ()


@dataclass(frozen=True)
class BayesianSamplerDiagnostics:
    """Compact sampler diagnostics for the Bayesian backend."""

    walker_count: int
    step_count: int
    mean_acceptance_fraction: float
    max_alignment_count: int


@dataclass(frozen=True)
class BayesianScoringSummary:
    """Serializable Bayesian scoring surface for nuisance-parameter TTV fits."""

    status: str
    contract_version: str
    sampler: str
    credible_interval: float
    rejection_log_bayes_factor_threshold: float
    observed_transit_count: int
    sample_count: int
    requested_sample_count: int
    warmup_draws: int
    nuisance_parameters: dict[str, BayesianPosteriorInterval]
    mass_limits: BayesianMassLimitCurve
    reference_solution: dict[str, float | None]
    diagnostics: BayesianSamplerDiagnostics | None = None
    posterior_samples: dict[str, list[float]] | None = None


def _serialize_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            field.name: _serialize_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize_value(item) for item in value]
    return value


def _deserialize_bayesian(data: dict) -> BayesianScoringSummary:
    nuisance_params = {
        k: BayesianPosteriorInterval(**v) for k, v in data["nuisance_parameters"].items()
    }
    
    posteriors = []
    for p in data["mass_limits"]["posterior_by_period_ratio"]:
        posteriors.append(BayesianMassPosterior(
            period_ratio=p["period_ratio"],
            masses=np.array(p["masses"]),
            log_evidence=np.array(p["log_evidence"]),
            model_probabilities=np.array(p["model_probabilities"]),
            cumulative_probability=np.array(p["cumulative_probability"]),
            posterior_predictive_score=np.array(p["posterior_predictive_score"]),
            best_mass=p["best_mass"],
            credible_upper_bound=p["credible_upper_bound"],
            rejection_upper_bound=p["rejection_upper_bound"],
            upper_bound=p["upper_bound"],
        ))
        
    mass_limits = BayesianMassLimitCurve(
        period_ratios=np.array(data["mass_limits"]["period_ratios"]),
        evaluated_masses=np.array(data["mass_limits"]["evaluated_masses"]),
        credible_upper_bound=tuple(data["mass_limits"]["credible_upper_bound"]),
        rejection_upper_bound=tuple(data["mass_limits"]["rejection_upper_bound"]),
        upper_bound=tuple(data["mass_limits"]["upper_bound"]),
        units=data["mass_limits"]["units"],
        posterior_by_period_ratio=tuple(posteriors),
    )
    
    diagnostics = None
    if data.get("diagnostics"):
        diagnostics = BayesianSamplerDiagnostics(**data["diagnostics"])
        
    return BayesianScoringSummary(
        status=data["status"],
        contract_version=data["contract_version"],
        sampler=data["sampler"],
        credible_interval=data["credible_interval"],
        rejection_log_bayes_factor_threshold=data["rejection_log_bayes_factor_threshold"],
        observed_transit_count=data["observed_transit_count"],
        sample_count=data["sample_count"],
        requested_sample_count=data["requested_sample_count"],
        warmup_draws=data["warmup_draws"],
        nuisance_parameters=nuisance_params,
        mass_limits=mass_limits,
        reference_solution=data["reference_solution"],
        diagnostics=diagnostics,
        posterior_samples=data.get("posterior_samples")
    )


@dataclass(frozen=True)
class MassThresholds:
    """Critical-mass curves derived from one scoring backend over a TTV grid."""

    chi2: np.ndarray
    rms: np.ndarray
    backend: str = DEFAULT_MASS_THRESHOLD_BACKEND
    chi2_threshold: float | None = None
    rms_threshold: float | None = None
    chi2_degrees_of_freedom: int | None = None
    chi2_surface: np.ndarray | None = None
    reduced_chi2_surface: np.ndarray | None = None
    relative_log_likelihood_surface: np.ndarray | None = None
    period_ratios: np.ndarray | None = None
    companion_masses: np.ndarray | None = None
    bayesian: BayesianScoringSummary | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend": self.backend,
            "chi2": self.chi2.tolist(),
            "rms": self.rms.tolist(),
            "chi2_threshold": self.chi2_threshold,
            "rms_threshold": self.rms_threshold,
        }
        if self.chi2_degrees_of_freedom is not None:
            payload["chi2_degrees_of_freedom"] = int(self.chi2_degrees_of_freedom)
        if self.chi2_surface is not None:
            payload["chi2_surface"] = np.asarray(self.chi2_surface).tolist()
        if self.reduced_chi2_surface is not None:
            payload["reduced_chi2_surface"] = np.asarray(
                self.reduced_chi2_surface
            ).tolist()
        if self.relative_log_likelihood_surface is not None:
            payload["relative_log_likelihood_surface"] = np.asarray(
                self.relative_log_likelihood_surface
            ).tolist()
        if self.period_ratios is not None:
            payload["period_ratios"] = np.asarray(self.period_ratios).tolist()
        if self.companion_masses is not None:
            payload["companion_masses"] = np.asarray(self.companion_masses).tolist()
        if self.bayesian is not None:
            payload["bayesian"] = _serialize_value(self.bayesian)
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "MassThresholds":
        kwargs = {
            "chi2": np.array(data["chi2"]),
            "rms": np.array(data["rms"]),
            "backend": data.get("backend", DEFAULT_MASS_THRESHOLD_BACKEND),
            "chi2_threshold": data.get("chi2_threshold"),
            "rms_threshold": data.get("rms_threshold"),
        }
        
        if "chi2_degrees_of_freedom" in data:
            kwargs["chi2_degrees_of_freedom"] = data["chi2_degrees_of_freedom"]
            
        for k in ["chi2_surface", "reduced_chi2_surface", "relative_log_likelihood_surface", "period_ratios", "companion_masses"]:
            if k in data and data[k] is not None:
                kwargs[k] = np.array(data[k])
                
        if data.get("bayesian"):
            kwargs["bayesian"] = _deserialize_bayesian(data["bayesian"])
            
        return cls(**kwargs)


class MassThresholdScorer(Protocol):
    """Protocol for backend objects that extract critical-mass curves."""

    def critical_masses(
        self,
        *,
        ttv_results,
        epoch,
        ttv_mcmc,
        ttv_err,
        period_ratios,
        companion_masses,
    ) -> MassThresholds: ...


def first_rejected_mass(score_2d, crit, *, valid_2d, period_ratios, companion_masses):
    """Return the first rejected companion mass for each period-ratio column."""

    rejected_masses = []
    for ratio_index, _ in enumerate(period_ratios):
        for mass_index, companion_mass in enumerate(companion_masses):
            if not valid_2d[mass_index, ratio_index]:
                continue
            if not np.isfinite(score_2d[mass_index, ratio_index]):
                continue
            if score_2d[mass_index, ratio_index] >= crit:
                rejected_masses.append(companion_mass)
                break
    return np.array(rejected_masses)


class Chi2AndRmsMassThresholdScorer:
    """Default scoring backend that preserves the current chi2/RMS behavior."""

    def critical_masses(
        self,
        *,
        ttv_results,
        epoch,
        ttv_mcmc,
        ttv_err,
        period_ratios,
        companion_masses,
    ) -> MassThresholds:
        chi2_degrees_of_freedom = int(len(ttv_mcmc))
        chi2 = get_chi2_v(
            ttv_rebound=np.array(ttv_results),
            epoch=epoch,
            ttv_mcmc=ttv_mcmc,
            ttv_err=ttv_err,
        )
        chi2_crit = scipy.stats.chi2.ppf(0.997, chi2_degrees_of_freedom)

        rms = get_rms_v(ttv_results)
        rms_crit = np.sqrt(np.mean(ttv_mcmc**2))

        chi2_2d = np.array(chi2).reshape(len(companion_masses), len(period_ratios))
        rms_2d = np.array(rms).reshape(len(companion_masses), len(period_ratios))
        reduced_chi2_2d = chi2_2d / chi2_degrees_of_freedom
        valid_2d = np.isfinite(chi2_2d) & np.isfinite(rms_2d)

        return MassThresholds(
            chi2=first_rejected_mass(
                chi2_2d,
                chi2_crit,
                valid_2d=valid_2d,
                period_ratios=period_ratios,
                companion_masses=companion_masses,
            ),
            rms=first_rejected_mass(
                rms_2d,
                rms_crit,
                valid_2d=valid_2d,
                period_ratios=period_ratios,
                companion_masses=companion_masses,
            ),
            backend=DEFAULT_MASS_THRESHOLD_BACKEND,
            chi2_threshold=chi2_crit,
            rms_threshold=rms_crit,
            chi2_degrees_of_freedom=chi2_degrees_of_freedom,
            chi2_surface=chi2_2d,
            reduced_chi2_surface=reduced_chi2_2d,
            relative_log_likelihood_surface=-0.5 * chi2_2d,
            period_ratios=np.asarray(period_ratios, dtype=float),
            companion_masses=np.asarray(companion_masses, dtype=float),
        )


@dataclass(frozen=True)
class _BayesianPriorScales:
    offset_sigma: float
    jitter_min: float
    jitter_max: float


@dataclass(frozen=True)
class _BayesianModelResult:
    support_score: float
    log_evidence: float
    sample_count: int
    intervals: dict[str, BayesianPosteriorInterval]
    mean_acceptance_fraction: float
    alignment_count: int
    posterior_samples: dict[str, list[float]] | None = None


def _alignment_count(ttv_rebound: np.ndarray, epoch: np.ndarray) -> int:
    span = int(epoch[-1] - epoch[0])
    alignment_count = int(len(ttv_rebound) - span)
    if alignment_count <= 0:
        raise ValueError("ttv_rebound must be long enough to support epoch alignment")
    return alignment_count


def _aligned_signal(ttv_rebound: np.ndarray, epoch: np.ndarray, shift_index: int) -> np.ndarray:
    indices = np.asarray(epoch - epoch[0] + shift_index, dtype=int)
    return np.asarray(ttv_rebound, dtype=float)[indices]


def _credible_interval(samples: np.ndarray, credible_interval: float) -> BayesianPosteriorInterval:
    if samples.size == 0:
        return BayesianPosteriorInterval()
    alpha = (1.0 - credible_interval) / 2.0
    lower, median, upper = np.quantile(samples, [alpha, 0.5, 1.0 - alpha])
    return BayesianPosteriorInterval(
        median=float(median),
        lower=float(lower),
        upper=float(upper),
    )


def _stable_seed(*parts) -> int:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            array = np.asarray(part)
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(np.ascontiguousarray(array).tobytes())
        else:
            digest.update(repr(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") % (2**32 - 1)


def _posterior_credible_upper_bound(
    model_probabilities: np.ndarray,
    masses: np.ndarray,
    credible_interval: float,
) -> float | None:
    cumulative_probability = np.cumsum(model_probabilities)
    for index, probability in enumerate(cumulative_probability):
        if probability >= credible_interval:
            if index == 0:
                return None
            return float(masses[index])
    return None


def _rejection_upper_bound(
    masses: np.ndarray,
    log_evidence: np.ndarray,
    *,
    rejection_log_bayes_factor_threshold: float,
) -> float | None:
    masses = np.asarray(masses, dtype=float)
    log_evidence = np.asarray(log_evidence, dtype=float)
    reference_index = int(np.argmax(log_evidence))
    if reference_index == 0:
        search_indices = range(1, len(masses))
    else:
        search_indices = range(reference_index + 1, len(masses))

    for index in search_indices:
        delta_log_evidence = float(log_evidence[index] - log_evidence[reference_index])
        if delta_log_evidence < rejection_log_bayes_factor_threshold:
            return float(masses[index])
    return None


def _relative_model_probabilities(log_evidences: list[float]) -> np.ndarray:
    evidence_array = np.asarray(log_evidences, dtype=float)
    evidence_array = evidence_array - evidence_array.max()
    probabilities = np.exp(evidence_array)
    return probabilities / probabilities.sum()


class BayesianMassThresholdScorer:
    """Experimental Bayesian posterior mass-summary backend with nuisance marginalization."""

    _JITTER_EVIDENCE_QUADRATURE_ORDER = 32

    def __init__(self, config=None):
        if config is None:
            from .config import BayesianScoringConfig

            config = BayesianScoringConfig()
        self.config = config

    def _validate_nuisance_parameters(self) -> tuple[str, ...]:
        unsupported = tuple(
            parameter
            for parameter in self.config.nuisance_parameters
            if parameter not in SUPPORTED_BAYESIAN_NUISANCE_PARAMETERS
        )
        if unsupported:
            raise ValueError(
                "Unsupported Bayesian nuisance parameters: " + ", ".join(unsupported)
            )
        return tuple(self.config.nuisance_parameters)

    @staticmethod
    def _representative_chain_rows(flat_chain: np.ndarray, sample_count: int) -> np.ndarray:
        if flat_chain.shape[0] < sample_count:
            raise RuntimeError("Bayesian sampler did not retain the requested sample count")
        if flat_chain.shape[0] == sample_count:
            return flat_chain

        row_count = flat_chain.shape[0]
        step = row_count / sample_count
        indices = np.floor((np.arange(sample_count, dtype=float) + 0.5) * step).astype(
            int
        )
        indices = np.clip(indices, 0, row_count - 1)
        return flat_chain[indices]

    def _prior_scales(
        self,
        observed_ttv: np.ndarray,
        aligned_signal: np.ndarray,
        ttv_err: np.ndarray,
    ) -> _BayesianPriorScales:
        scale = max(
            float(np.std(observed_ttv)),
            float(np.std(aligned_signal)),
            float(np.mean(ttv_err)),
            1e-3,
        )
        return _BayesianPriorScales(
            offset_sigma=max(5.0 * scale, float(np.mean(ttv_err))),
            jitter_min=max(float(np.min(ttv_err)) * 1e-3, 1e-6),
            jitter_max=max(10.0 * scale, float(np.max(ttv_err))),
        )

    @staticmethod
    def _log_gaussian_likelihood(
        residual: np.ndarray,
        sigma_squared: np.ndarray,
        *,
        offset_sigma: float | None,
    ) -> float:
        inverse_sigma_squared = 1.0 / sigma_squared
        logdet = float(np.sum(np.log(sigma_squared)))
        quadratic = float(np.sum(residual**2 * inverse_sigma_squared))

        if offset_sigma is not None:
            offset_variance = float(offset_sigma**2)
            weighted_residual_sum = float(np.sum(residual * inverse_sigma_squared))
            precision_sum = float(np.sum(inverse_sigma_squared))
            determinant_correction = 1.0 + offset_variance * precision_sum
            quadratic -= (offset_variance * weighted_residual_sum**2) / determinant_correction
            logdet += float(np.log(determinant_correction))

        sample_count = residual.size
        return float(-0.5 * (sample_count * np.log(2.0 * np.pi) + logdet + quadratic))

    def _log_evidence(
        self,
        *,
        ttv_rebound: np.ndarray,
        epoch: np.ndarray,
        observed_ttv: np.ndarray,
        observed_err: np.ndarray,
        nuisance_parameters: tuple[str, ...],
        prior_scales: _BayesianPriorScales,
        alignment_count: int,
    ) -> float:
        """Return log evidence under exact epoch marginalization and log-jitter integration.

        When `jitter` is active, the integration variable is `log_jitter`, so the
        evidence uses a log-uniform prior between `prior_scales.jitter_min` and
        `prior_scales.jitter_max`.
        """

        if "epoch_shift" in nuisance_parameters:
            shift_indices = range(alignment_count)
        else:
            shift_indices = (0,)

        offset_sigma = prior_scales.offset_sigma if "baseline_offset" in nuisance_parameters else None
        use_jitter = "jitter" in nuisance_parameters

        jitter_log_nodes = np.array([0.0], dtype=float)
        jitter_log_weights = np.array([0.0], dtype=float)
        if use_jitter:
            log_jitter_min = float(np.log(prior_scales.jitter_min))
            log_jitter_max = float(np.log(prior_scales.jitter_max))
            quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
                self._JITTER_EVIDENCE_QUADRATURE_ORDER
            )
            jitter_log_nodes = 0.5 * (log_jitter_max - log_jitter_min) * quadrature_nodes + 0.5 * (
                log_jitter_max + log_jitter_min
            )
            jitter_log_weights = np.log(0.5 * quadrature_weights)

        shift_log_evidences: list[float] = []
        for shift_index in shift_indices:
            residual = observed_ttv - _aligned_signal(ttv_rebound, epoch, shift_index)
            if not use_jitter:
                shift_log_evidences.append(
                    self._log_gaussian_likelihood(
                        residual,
                        observed_err**2,
                        offset_sigma=offset_sigma,
                    )
                )
                continue

            jitter_terms = []
            for log_jitter, log_weight in zip(jitter_log_nodes, jitter_log_weights, strict=True):
                jitter_terms.append(
                    self._log_gaussian_likelihood(
                        residual,
                        observed_err**2 + np.exp(2.0 * log_jitter),
                        offset_sigma=offset_sigma,
                    )
                    + float(log_weight)
                )
            shift_log_evidences.append(float(logsumexp(jitter_terms)))

        if "epoch_shift" in nuisance_parameters:
            return float(logsumexp(shift_log_evidences) - np.log(len(shift_log_evidences)))
        return shift_log_evidences[0]

    def _sample_model(
        self,
        *,
        ttv_rebound: np.ndarray,
        epoch: np.ndarray,
        observed_ttv: np.ndarray,
        observed_err: np.ndarray,
        nuisance_parameters: tuple[str, ...],
        seed_hint,
    ) -> _BayesianModelResult:
        if not np.all(np.isfinite(ttv_rebound)):
            return _BayesianModelResult(
                support_score=INVALID_MODEL_LOG_SCORE,
                log_evidence=INVALID_MODEL_LOG_SCORE,
                sample_count=0,
                intervals={
                    name: BayesianPosteriorInterval()
                    for name in nuisance_parameters
                },
                mean_acceptance_fraction=0.0,
                alignment_count=0,
                posterior_samples=None,
            )
        alignment_count = _alignment_count(ttv_rebound, epoch)
        aligned_reference = _aligned_signal(ttv_rebound, epoch, 0)
        prior_scales = self._prior_scales(observed_ttv, aligned_reference, observed_err)
        log_evidence = self._log_evidence(
            ttv_rebound=ttv_rebound,
            epoch=epoch,
            observed_ttv=observed_ttv,
            observed_err=observed_err,
            nuisance_parameters=nuisance_parameters,
            prior_scales=prior_scales,
            alignment_count=alignment_count,
        )

        parameter_names: list[str] = []
        if "epoch_shift" in nuisance_parameters:
            parameter_names.append("epoch_shift")
        if "baseline_offset" in nuisance_parameters:
            parameter_names.append("baseline_offset")
        if "jitter" in nuisance_parameters:
            parameter_names.append("log_jitter")

        ndim = len(parameter_names)
        if ndim == 0:
            raise ValueError("Bayesian scorer requires at least one nuisance parameter")

        name_to_index = {name: index for index, name in enumerate(parameter_names)}
        jitter_log_min = float(np.log(prior_scales.jitter_min))
        jitter_log_max = float(np.log(prior_scales.jitter_max))

        def unpack(theta: np.ndarray) -> tuple[int, float, float]:
            shift_index = 0
            if "epoch_shift" in name_to_index:
                # `epoch_shift` stays discrete in the evidence calculation above.
                # The sampler only uses this rounded latent representation for
                # posterior diagnostics, so it should not be interpreted as a
                # continuous physical parameter.
                shift_raw = theta[name_to_index["epoch_shift"]]
                if shift_raw < -0.49 or shift_raw > alignment_count - 0.51:
                    raise ValueError
                shift_index = int(np.clip(np.rint(shift_raw), 0, alignment_count - 1))

            offset = 0.0
            if "baseline_offset" in name_to_index:
                offset = float(theta[name_to_index["baseline_offset"]])

            jitter = 0.0
            if "log_jitter" in name_to_index:
                log_jitter = float(theta[name_to_index["log_jitter"]])
                if log_jitter < jitter_log_min or log_jitter > jitter_log_max:
                    raise ValueError
                jitter = float(np.exp(log_jitter))
            return shift_index, offset, jitter

        def log_probability(theta: np.ndarray) -> float:
            try:
                shift_index, offset, jitter = unpack(theta)
            except ValueError:
                return -np.inf

            model_ttv = _aligned_signal(ttv_rebound, epoch, shift_index) + offset
            sigma = np.sqrt(observed_err**2 + jitter**2)
            residual = observed_ttv - model_ttv
            log_likelihood = -0.5 * np.sum(
                (residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2)
            )

            log_prior = 0.0
            if "baseline_offset" in name_to_index:
                log_prior -= 0.5 * (offset / prior_scales.offset_sigma) ** 2
                log_prior -= np.log(np.sqrt(2.0 * np.pi) * prior_scales.offset_sigma)
            return float(log_likelihood + log_prior)

        nwalkers = max(12, 4 * ndim)
        posterior_steps = int(np.ceil(self.config.posterior_sample_count / nwalkers))
        nsteps = self.config.warmup_draws + posterior_steps

        seed = _stable_seed(
            observed_ttv,
            observed_err,
            ttv_rebound,
            nuisance_parameters,
            self.config.credible_interval,
            self.config.posterior_sample_count,
            self.config.warmup_draws,
            seed_hint,
        )
        rng = np.random.default_rng(seed)
        initial_position = np.empty((nwalkers, ndim), dtype=float)
        if "epoch_shift" in name_to_index:
            index = name_to_index["epoch_shift"]
            initial_position[:, index] = rng.integers(0, alignment_count, size=nwalkers)
            initial_position[:, index] += rng.normal(0.0, 0.05, size=nwalkers)
            initial_position[:, index] = np.clip(
                initial_position[:, index], -0.45, alignment_count - 0.55
            )
        if "baseline_offset" in name_to_index:
            index = name_to_index["baseline_offset"]
            residual_mean = float(np.mean(observed_ttv - aligned_reference))
            initial_position[:, index] = residual_mean + rng.normal(
                0.0,
                prior_scales.offset_sigma * 0.1,
                size=nwalkers,
            )
            initial_position[:, index] = np.clip(
                initial_position[:, index],
                -5.5 * prior_scales.offset_sigma,
                5.5 * prior_scales.offset_sigma,
            )
        if "log_jitter" in name_to_index:
            index = name_to_index["log_jitter"]
            reference_jitter = np.clip(
                float(np.median(observed_err)) * 0.5,
                prior_scales.jitter_min * 1.01,
                prior_scales.jitter_max * 0.99,
            )
            initial_position[:, index] = np.log(reference_jitter) + rng.normal(
                0.0, 0.2, size=nwalkers
            )
            initial_position[:, index] = np.clip(
                initial_position[:, index],
                jitter_log_min + 1e-6,
                jitter_log_max - 1e-6,
            )

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.random_state = np.random.RandomState(seed).get_state()
        sampler.run_mcmc(initial_position, nsteps, progress=False)

        flat_chain = sampler.get_chain(discard=self.config.warmup_draws, flat=True)
        retained_chain = self._representative_chain_rows(
            flat_chain, self.config.posterior_sample_count
        )

        posterior_samples: dict[str, np.ndarray] = {}
        if "epoch_shift" in nuisance_parameters:
            posterior_samples["epoch_shift"] = np.clip(
                np.rint(retained_chain[:, name_to_index["epoch_shift"]]),
                0,
                alignment_count - 1,
            ).astype(float)
        if "baseline_offset" in nuisance_parameters:
            posterior_samples["baseline_offset"] = retained_chain[
                :, name_to_index["baseline_offset"]
            ]
        if "jitter" in nuisance_parameters:
            posterior_samples["jitter"] = np.exp(
                retained_chain[:, name_to_index["log_jitter"]]
            )

        shift_samples = posterior_samples.get(
            "epoch_shift",
            np.zeros(self.config.posterior_sample_count, dtype=float),
        )
        offset_samples = posterior_samples.get(
            "baseline_offset",
            np.zeros(self.config.posterior_sample_count, dtype=float),
        )
        jitter_samples = posterior_samples.get(
            "jitter",
            np.zeros(self.config.posterior_sample_count, dtype=float),
        )

        pointwise_log_likelihood = []
        for sample_index in range(self.config.posterior_sample_count):
            shift_index = int(shift_samples[sample_index])
            offset = float(offset_samples[sample_index])
            jitter = float(jitter_samples[sample_index])
            model_ttv = _aligned_signal(ttv_rebound, epoch, shift_index) + offset
            sigma = np.sqrt(observed_err**2 + jitter**2)
            pointwise_log_likelihood.append(
                -0.5 * (((observed_ttv - model_ttv) / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
            )
        pointwise_log_likelihood = np.asarray(pointwise_log_likelihood)
        support_score = float(
            logsumexp(pointwise_log_likelihood.sum(axis=1))
            - np.log(pointwise_log_likelihood.shape[0])
        )

        intervals = {
            name: _credible_interval(posterior_samples[name], self.config.credible_interval)
            for name in nuisance_parameters
        }

        chains_payload = None
        if self.config.store_chains:
            chains_payload = {
                name: posterior_samples[name].astype(float).tolist()
                for name in nuisance_parameters
            }

        return _BayesianModelResult(
            support_score=support_score,
            log_evidence=log_evidence,
            sample_count=self.config.posterior_sample_count,
            intervals=intervals,
            mean_acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
            alignment_count=alignment_count,
            posterior_samples=chains_payload,
        )

    def critical_masses(
        self,
        *,
        ttv_results,
        epoch,
        ttv_mcmc,
        ttv_err,
        period_ratios,
        companion_masses,
    ) -> MassThresholds:
        nuisance_parameters = self._validate_nuisance_parameters()
        epoch = np.asarray(epoch, dtype=int)
        observed_ttv = np.asarray(ttv_mcmc, dtype=float)
        observed_err = np.asarray(ttv_err, dtype=float)
        period_ratios = np.asarray(period_ratios, dtype=float)
        companion_masses = np.asarray(companion_masses, dtype=float)
        ttv_grid = [np.asarray(ttv_result, dtype=float) for ttv_result in ttv_results]

        expected_grid_size = len(period_ratios) * len(companion_masses)
        if len(ttv_grid) != expected_grid_size:
            raise ValueError("ttv_results must match the configured mass-ratio grid size")

        zero_signal = np.zeros_like(ttv_grid[0], dtype=float)
        null_result = self._sample_model(
            ttv_rebound=zero_signal,
            epoch=epoch,
            observed_ttv=observed_ttv,
            observed_err=observed_err,
            nuisance_parameters=nuisance_parameters,
            seed_hint=("null",),
        )

        model_results = []
        for model_index, ttv_rebound in enumerate(ttv_grid):
            mass_index = model_index // len(period_ratios)
            ratio_index = model_index % len(period_ratios)
            model_results.append(
                self._sample_model(
                    ttv_rebound=ttv_rebound,
                    epoch=epoch,
                    observed_ttv=observed_ttv,
                    observed_err=observed_err,
                    nuisance_parameters=nuisance_parameters,
                    seed_hint=(float(period_ratios[ratio_index]), float(companion_masses[mass_index])),
                )
            )

        posterior_by_ratio: list[BayesianMassPosterior] = []
        credible_upper_bounds: list[float | None] = []
        rejection_upper_bounds: list[float | None] = []
        reference_intervals = null_result.intervals
        reference_samples = null_result.posterior_samples
        reference_solution = {"period_ratio": None, "companion_mass": None}
        reference_log_evidence = null_result.log_evidence

        for ratio_index, period_ratio in enumerate(period_ratios):
            ratio_results = [
                model_results[mass_index * len(period_ratios) + ratio_index]
                for mass_index in range(len(companion_masses))
            ]
            log_evidences = [null_result.log_evidence] + [
                result.log_evidence for result in ratio_results
            ]
            support_scores = [null_result.support_score] + [
                result.support_score for result in ratio_results
            ]
            model_probabilities = _relative_model_probabilities(log_evidences)
            masses = np.concatenate(([0.0], companion_masses))
            cumulative_probability = np.cumsum(model_probabilities)
            best_model_index = int(np.argmax(model_probabilities))
            best_mass = None if best_model_index == 0 else float(masses[best_model_index])
            credible_upper_bound = _posterior_credible_upper_bound(
                model_probabilities,
                masses,
                self.config.credible_interval,
            )
            rejection_upper_bound = _rejection_upper_bound(
                masses,
                np.asarray(log_evidences, dtype=float),
                rejection_log_bayes_factor_threshold=(
                    self.config.rejection_log_bayes_factor_threshold
                ),
            )
            credible_upper_bounds.append(credible_upper_bound)
            rejection_upper_bounds.append(rejection_upper_bound)
            posterior_by_ratio.append(
                BayesianMassPosterior(
                    period_ratio=float(period_ratio),
                    masses=masses,
                    log_evidence=np.asarray(log_evidences, dtype=float),
                    model_probabilities=model_probabilities,
                    cumulative_probability=cumulative_probability,
                    posterior_predictive_score=np.asarray(support_scores, dtype=float),
                    best_mass=best_mass,
                    credible_upper_bound=credible_upper_bound,
                    rejection_upper_bound=rejection_upper_bound,
                    upper_bound=credible_upper_bound,
                )
            )

            if best_model_index != 0:
                best_result = ratio_results[best_model_index - 1]
                if best_result.log_evidence > reference_log_evidence:
                    reference_log_evidence = best_result.log_evidence
                    reference_intervals = best_result.intervals
                    reference_samples = best_result.posterior_samples
                    reference_solution = {
                        "period_ratio": float(period_ratio),
                        "companion_mass": float(best_mass),
                    }

        diagnostics = BayesianSamplerDiagnostics(
            walker_count=max(12, 4 * len(nuisance_parameters)),
            step_count=self.config.warmup_draws
            + int(np.ceil(self.config.posterior_sample_count / max(12, 4 * len(nuisance_parameters)))),
            mean_acceptance_fraction=float(
                np.mean([null_result.mean_acceptance_fraction] + [result.mean_acceptance_fraction for result in model_results])
            ),
            max_alignment_count=max(
                [null_result.alignment_count] + [result.alignment_count for result in model_results]
            ),
        )

        return MassThresholds(
            chi2=np.array([], dtype=float),
            rms=np.array([], dtype=float),
            backend=BAYESIAN_MASS_THRESHOLD_BACKEND,
            bayesian=BayesianScoringSummary(
                status="posterior_sampled",
                contract_version="stage4_phase2",
                sampler="emcee",
                credible_interval=self.config.credible_interval,
                rejection_log_bayes_factor_threshold=(
                    self.config.rejection_log_bayes_factor_threshold
                ),
                observed_transit_count=len(observed_ttv),
                sample_count=self.config.posterior_sample_count,
                requested_sample_count=self.config.posterior_sample_count,
                warmup_draws=self.config.warmup_draws,
                nuisance_parameters=reference_intervals,
                mass_limits=BayesianMassLimitCurve(
                    period_ratios=period_ratios,
                    evaluated_masses=companion_masses,
                    credible_upper_bound=tuple(credible_upper_bounds),
                    rejection_upper_bound=tuple(rejection_upper_bounds),
                    upper_bound=tuple(credible_upper_bounds),
                    posterior_by_period_ratio=tuple(posterior_by_ratio),
                ),
                reference_solution=reference_solution,
                diagnostics=diagnostics,
                posterior_samples=reference_samples,
            ),
        )


MASS_THRESHOLD_SCORER_REGISTRY = {
    DEFAULT_MASS_THRESHOLD_BACKEND: Chi2AndRmsMassThresholdScorer,
    BAYESIAN_MASS_THRESHOLD_BACKEND: BayesianMassThresholdScorer,
}


def make_mass_threshold_scorer(backend: str, *, bayesian_config=None) -> MassThresholdScorer:
    """Build a supported mass-threshold scorer from a typed backend name."""

    if backend == BAYESIAN_MASS_THRESHOLD_BACKEND:
        return BayesianMassThresholdScorer(config=bayesian_config)
    try:
        return MASS_THRESHOLD_SCORER_REGISTRY[backend]()
    except KeyError as error:
        raise ValueError(f"Unsupported scoring backend: {backend}") from error


def supported_mass_threshold_backends() -> tuple[str, ...]:
    """Return the backend names currently supported by the scorer factory."""

    return tuple(MASS_THRESHOLD_SCORER_REGISTRY)


get_chi2_v = np.vectorize(
    get_chi2,
    excluded=["epoch", "ttv_mcmc", "ttv_err"],
    signature="(n)->()",
)
get_rms_v = np.vectorize(get_rms, signature="(n)->()")
