from pytransit.lpf.tesslpf import BaseLPF, fold, downsample_time
import numpy as np
from matplotlib import pyplot as plt
from corner import corner


'''
This class, SingleFit, is an extension of the pytransit.BaseLPF class and is
used for single transit fitting. It provides methods to plot single transits
and single folded transits, to see the quality of the fit. It also provides a
corner plot which helps visualize the posterior samples for individual
parameters using the posterion_samples() method.
'''


class SingleFit(BaseLPF):
    def plot_single_transit(
        self,
        method="de",
        figsize=(13, 6),
        ylim=(0.9975, 1.002),
        xlim=None,
        binwidth=8,
        remove_baseline: bool = False,
    ):
        if method == "de":
            pv = self.de.minimum_location
            tc, p = pv[[0, 1]]
        else:
            raise NotImplementedError

        phase = p * fold(self.timea, p, tc, 0.5)
        binwidth = binwidth / 24 / 60
        sids = np.argsort(phase)

        tm = self.transit_model(pv)

        if remove_baseline:
            gp = self._lnlikelihood_models[0]
            bl = np.squeeze(gp.predict_baseline(pv))
        else:
            bl = np.ones_like(self.ofluxa)

        bp, bfo, beo = downsample_time(phase[sids], (self.ofluxa / bl)[sids], binwidth)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(phase - 0.5 * p, self.ofluxa / bl, ".", alpha=0.15)
        ax.errorbar(bp - 0.5 * p, bfo, beo, fmt="ko")
        ax.plot(phase[sids] - 0.5 * p, tm[sids], "k")
        xlim = (
            xlim
            if xlim is not None
            else 1.01 * (bp[np.isfinite(bp)][[0, -1]] - 0.5 * p)
        )
        plt.setp(
            ax, ylim=ylim, xlim=xlim, xlabel="Time - Tc [d]", ylabel="Normalised flux"
        )
        fig.tight_layout()
        return fig

    def plot_single_folded_transit(
        self, figsize=(13, 6), ylim=(0.9975, 1.002), xlim=None
    ):
        if method == "de":
            pv = self.de.minimum_location
            tc, p = pv[[0, 1]]
        else:
            raise NotImplementedError

        phase = p * fold(self.timea, p, tc, 0.5)
        binwidth = binwidth / 24 / 60
        sids = np.argsort(phase)

        tm = self.transit_model(pv)

        if remove_baseline:
            gp = self._lnlikelihood_models[0]
            bl = np.squeeze(gp.predict_baseline(pv))
        else:
            bl = np.ones_like(self.ofluxa)

        bp, bfo, beo = downsample_time(phase[sids], (self.ofluxa / bl)[sids], binwidth)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(phase - 0.5 * p, self.ofluxa / bl, ".", alpha=0.15)
        ax.errorbar(bp - 0.5 * p, bfo, beo, fmt="ko")
        ax.plot(phase[sids] - 0.5 * p, tm[sids], "k")
        xlim = (
            xlim
            if xlim is not None
            else 1.01 * (bp[np.isfinite(bp)][[0, -1]] - 0.5 * p)
        )
        plt.setp(
            ax, ylim=ylim, xlim=xlim, xlabel="Time - Tc [d]", ylabel="Normalised flux"
        )
        fig.tight_layout()
        return fig

    def plot_corner(self):
        fig, ax = plt.subplots(5, 5, figsize=(10, 10), dpi=200)

        df = self.posterior_samples()
        corner(
            df["tc p rho b k".split()],
            labels="\n\n\nZero epoch, \n\n\nPeriod, \n\n\nStellar density, \n\n\nimpact parameter, \n\n\nradius ratio".split(
                ", "
            ),
            fig=fig,
            show_titles=True,
        )

        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        return fig
