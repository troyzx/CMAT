import json
from datetime import datetime, timezone
import numpy as np

CACHE_SCHEMA_VERSION = "1"


def _save_npz(path, **kwargs):
    kwargs["cache_schema_version"] = np.array([CACHE_SCHEMA_VERSION])
    kwargs["created_at"] = np.array([datetime.now(timezone.utc).isoformat()])
    np.savez_compressed(path, **kwargs)


def save_ttv_grid(path, *, period_ratios, companion_masses, epochs, ttv_mcmc, ttv_err, ttv_results):
    _save_npz(
        path,
        period_ratios=np.asarray(period_ratios, dtype=float),
        companion_masses=np.asarray(companion_masses, dtype=float),
        epochs=np.asarray(epochs, dtype=int),
        ttv_mcmc=np.asarray(ttv_mcmc, dtype=float),
        ttv_err=np.asarray(ttv_err, dtype=float),
        ttv_results=np.asarray(ttv_results, dtype=float),
        grid_shape=np.array([len(companion_masses), len(period_ratios)])
    )


def load_ttv_grid(path):
    with np.load(path, allow_pickle=False) as data:
        schema_version = data["cache_schema_version"][0]
        if schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported cache schema version: {schema_version}")
        
        return {
            "period_ratios": data["period_ratios"],
            "companion_masses": data["companion_masses"],
            "epochs": data["epochs"],
            "ttv_mcmc": data["ttv_mcmc"],
            "ttv_err": data["ttv_err"],
            "ttv_results": data["ttv_results"],
        }


def save_megno_grid(path, *, period_ratios, companion_masses, megno_results):
    _save_npz(
        path,
        period_ratios=np.asarray(period_ratios, dtype=float),
        companion_masses=np.asarray(companion_masses, dtype=float),
        megno_results=np.asarray(megno_results, dtype=float),
        grid_shape=np.array([len(companion_masses), len(period_ratios)])
    )


def load_megno_grid(path):
    with np.load(path, allow_pickle=False) as data:
        schema_version = data["cache_schema_version"][0]
        if schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported cache schema version: {schema_version}")
        
        return {
            "period_ratios": data["period_ratios"],
            "companion_masses": data["companion_masses"],
            "megno_results": data["megno_results"],
        }


def save_scoring_summary(path, *, mass_thresholds):
    summary_dict = mass_thresholds.to_dict()
    summary_json = json.dumps(summary_dict)
    _save_npz(
        path,
        scoring_summary_json=np.array([summary_json])
    )


def load_scoring_summary(path):
    with np.load(path, allow_pickle=False) as data:
        schema_version = data["cache_schema_version"][0]
        if schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported cache schema version: {schema_version}")
        
        summary_json = data["scoring_summary_json"][0]
        return json.loads(summary_json)


def validate_ttv_grid_compatibility(cached, *, period_ratios, companion_masses, epochs, ttv_mcmc, ttv_err):
    if not np.allclose(cached["period_ratios"], period_ratios):
        raise ValueError("Cache period_ratios mismatch")
    if not np.allclose(cached["companion_masses"], companion_masses):
        raise ValueError("Cache companion_masses mismatch")
    if not np.array_equal(cached["epochs"], epochs):
        raise ValueError("Cache epochs mismatch")
    if not np.allclose(cached["ttv_mcmc"], ttv_mcmc):
        raise ValueError("Cache ttv_mcmc mismatch")
    if not np.allclose(cached["ttv_err"], ttv_err):
        raise ValueError("Cache ttv_err mismatch")
    if len(cached["ttv_results"]) != len(companion_masses) * len(period_ratios):
        raise ValueError("Cache ttv_results length mismatch")


def validate_megno_grid_compatibility(cached, *, period_ratios, companion_masses):
    if not np.allclose(cached["period_ratios"], period_ratios):
        raise ValueError("Cache period_ratios mismatch")
    if not np.allclose(cached["companion_masses"], companion_masses):
        raise ValueError("Cache companion_masses mismatch")
    if len(cached["megno_results"]) != len(companion_masses) * len(period_ratios):
        raise ValueError("Cache megno_results length mismatch")
