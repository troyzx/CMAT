"""Scoring helpers for comparing simulated and observed TTV residuals."""

import numpy as np


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


get_chi2_v = np.vectorize(
    get_chi2,
    excluded=["epoch", "ttv_mcmc", "ttv_err"],
    signature="(n)->()",
)
get_rms_v = np.vectorize(get_rms, signature="(n)->()")
