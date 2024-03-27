"""
This module contains a class `fitlpf` that performs a transit fit to a light
curve using the pytransit package. It also includes functions for downloading
data from the MAST archive, fitting individual transits,
and calculating the transit timing variations (TTVs) of the planet.

Functions:
- get_id(planet_name: str) -> int:
    Given the name of a planet, returns its TESS ID.
- get_prop(planet_name: str, tic: int) -> dict:
    Given the name of a planet and its TESS ID,
    returns a dictionary of its properties.
- truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100)
    -> matplotlib.colors.LinearSegmentedColormap:
    Truncates a colormap to a specified range.
- save_df_data(dir_path, file_name, df_data):
    Creates a new directory and saves data, while also checking if the folder
    exists and asking the user if they want to create a new folder,
    and if the file already exists, asking if the user wants to overwrite it.
- read_data(name: str) -> file:
    Reads data from a file.
- getn(unfloat) -> float:
    Given an `uncertainties.ufloat` object, returns its nominal value.
- gets(unfloat) -> float:
    Given an `uncertainties.ufloat` object, returns its standard deviation.
- epoch_v = np.vectorize(epoch) -> np.vectorize:
    Vectorizes the `epoch` function.
- fitlpf:
    A class that performs a transit fit to a light curve using the
    pytransit package. It also includes functions for downloading data
    from the MAST archive, fitting individual transits, and calculating
    the transit timing variations (TTVs) of the planet.
    Methods:
    - __init__(self, planet_name: str, datadir=None):
        Initializes the `fitlpf` object.
    - get_parameter(self):
        Gets the parameters of the planet.
    - print_parameters(self):
        Prints the parameters of the planet.
    - download_data(self) -> astropy.table.table.Table:
        Downloads data from the MAST archive.
    - de(self, niter=200, npop=30, datadir=None):
        Performs a differential evolution fit to the light curve.
    - plot_original_data(self) -> matplotlib.figure.Figure:
        Plots the original light curve.
    - fit_single(self, i, niter=100, npop=50, mcmc_repeats=4) -> SingleFit:
        Fits a single transit.
    - fit_singles(self, niter=100, npop=50):
        Fits all transits.
    - get_posterior_samples(self):
        Gets the posterior samples of the transit fits.
    - calculate_ttv(self):
        Calculates the transit timing variations (TTVs) of the planet.
    - plot_tcs(self, plot_zero_epoch=False):
        Plots the transit centers.
    - plot_ttv_re(
        self,
        plot_zero_epoch=False,
        set_epoch_zero=False,
        remove_baseline=True
    ):
        Plots the transit timing variations (TTVs) of the planet.
"""

import os
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astroquery.mast import Observations

# import rebound

# import scipy.stats
# from multiprocessing import get_context
# from tqdm.auto import tqdm
from uncertainties import ufloat
from pytransit.lpf.tesslpf import TESSLPF
from pytransit.orbits import epoch
from scipy.optimize import curve_fit
from cmat.singlefit import SingleFit

PLANETURL = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/"
DVURL = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/"
URL = PLANETURL + "/identifiers/"
header = {}

MJ_TO_MS = 9.5e-4
ME_TO_MS = 3.0e-6
RJ_TO_RS = 0.102792236
RS_TO_AU = 0.00464913034
DAY_TO_SEC = 24 * 60 * 60


def get_id(planet_name: str):
    """
    Get the TIC ID for a given planet name.

    Args:
        planet_name (str): The name of the planet.

    Returns:
        int: The TIC ID of the planet.
    """
    myparams = {"name": planet_name}
    r = requests.get(url=URL, params=myparams, headers=header, timeout=10)
    planet_names = r.json()
    ticid = planet_names["tessID"]

    return ticid


def get_prop(planet_name: str):
    """
    Get the properties of a planet.

    Args:
        planet_name (str): The name of the planet.

    Returns:
        dict: The properties of the planet in JSON format.
    """
    url = PLANETURL + planet_name + "/properties/"
    r = requests.get(url=url, headers=header, timeout=10)
    return r.json()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a colormap to a specified range.
    """

    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def save_df_data(dir_path, file_name, df_data):
    """
    create a new directory and save data, while also checking if the folder
    exists and asking the user if they want to create a new folder, and if the
    file already exists, asking if the user wants to overwrite it
    """
    # Check if the directory exists:
    if not os.path.exists(dir_path):
        # The directory doesn't exist, so ask the user if they want to create
        # it:
        create_dir = input(
            "The directory doesn't exist. Would you like to create it? (y/n) "
        )
        if create_dir.lower() == "y":
            os.makedirs(dir_path)
        else:
            print("Unable to save data.")
            exit()

    # Check if the file already exists:
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        # The file already exists, so ask the user if they want to overwrite
        # it:
        overwrite_file = input(
            "The file already exists. Do you want to overwrite it? (y/n) "
        )
        if overwrite_file.lower() != "y":
            print("Unable to save data.")
            exit()

    # Save some example data to the file:
    df_data.to_csv(file_path, index=False)

    print("Data saved successfully!")


def read_data(name: str):
    """
    Read data from a file.

    Args:
        name (str): The name of the file to read.

    Returns:
        file: The opened file object.

    """
    with open(name, "r", encoding="utf-8") as file:
        return file


def getn(unfloat):
    """
    Get the value of 'n' from the given 'unfloat' object.

    Parameters:
    unfloat (Unfloat): The Unfloat object.

    Returns:
    int: The value of 'n'.
    """
    return unfloat.n


def gets(unfloat):
    """
    Get the 's' attribute from the input unfloat object.

    Parameters:
    unfloat (object): The input unfloat object.

    Returns:
    str: The value of the 's' attribute.
    """
    return unfloat.s


getn_v = np.vectorize(getn)
gets_v = np.vectorize(gets)
epoch_v = np.vectorize(epoch)


class Fitlpf:
    """
    Class for fitting a TESS light curve with a transit model and
    calculating the transit timing variations (TTVs).

    Args:
        planet_name (str): Name of the planet.
        datadir (str, optional): Directory to store the downloaded data.
        Defaults to None.

    Attributes:
        planet_name (str): Name of the planet.
        ticid (int): TIC ID of the planet.
        period (uncertainties.core.Variable): Orbital period of the planet.
        zero_epoch (uncertainties.core.Variable): Zero epoch of the planet.
        prop (list): List of planet properties.
        datadir (str): Directory to store the downloaded data.
        lpf (TESSLPF): TESSLPF object for the planet.
        singles (list): List of SingleFit objects for each transit.
        post_samples (list): List of posterior samples for each transit.
        tcs (list): List of transit centers for each transit.
        epochs (list): List of epochs for each transit.

    Methods:
        get_parameter(): Get the planet parameters.
        print_parameters(): Print the planet parameters.
        download_data(): Download the TESS data for the planet.
        de(niter=200, npop=30, datadir=None): Differential evolution
        optimization for the planet.
        plot_original_data(): Plot the original TESS data for the planet.
        fit_single(i, niter=100, npop=50, mcmc_repeats=4): Fit a single
        transit.
        fit_singles(niter=100, npop=50): Fit all the transits.
        get_posterior_samples(): Get the posterior samples for all the
        transits.
        calculate_ttv(): Calculate the transit timing variations (TTVs).
        plot_tcs(plot_zero_epoch=False): Plot the transit centers.
        plot_ttv_re(
            plot_zero_epoch=False,
            set_epoch_zero=False,
            remove_baseline=True
            ):
            Plot the transit timing variations (TTVs).
    """

    def __init__(self, planet_name: str, datadir=None):
        if datadir is None:
            datadir = "./data/"
        self.full_datadir = datadir + planet_name
        self.planet_name = planet_name
        self.ticid = None
        self.period = None
        self.zero_epoch = None
        self.prop = []
        self.datadir = datadir
        self.lpf = None
        self.singles = []
        self.post_samples = []
        self.tcs = []
        self.epochs = []
        self.ttv_err = None
        self.ttv_mcmc_raw = None
        self.ttv_mcmc = None
        self.fit_single_v = None

    def get_parameter(self):
        """
        Retrieves the parameters for the given planet.

        Returns:
            None
        """
        planet_name = self.planet_name
        ticid = get_id(planet_name)
        self.prop = get_prop(planet_name)
        transit_time = self.prop[0]["transit_time"] + 2.4e6 + 0.5
        transit_time_err = max(
            self.prop[0]["transit_time_lower"], self.prop[0]["transit_time_upper"]
        )
        orbital_period = self.prop[0]["orbital_period"]
        orbital_period_err = max(
            self.prop[0]["orbital_period_lower"], self.prop[0]["orbital_period_upper"]
        )
        self.period = ufloat(orbital_period, orbital_period_err)
        self.zero_epoch = ufloat(transit_time, transit_time_err)
        self.ticid = ticid
        self.print_parameters()

    def print_parameters(self):
        """
        Print the parameters of the planet.

        This method prints the properties of the planet,
        including the stellar mass, planet mass,
        planet orbital period, transit time, and planet mass reference.

        Parameters:
            None

        Returns:
            None
        """
        planet_prop = self.prop
        print(f"{self.planet_name} Properties")
        print(
            f"Stellar Mass \t\t{planet_prop[0]['Ms']} \t\t"
            f"{planet_prop[0]['Ms_unit']}"
        )
        print(
            f"Planet Mass \t\t{planet_prop[0]['Mp']} \t\t"
            f"{planet_prop[0]['Mp_unit']}"
        )
        print(
            f"Planet Orbital Period \t{planet_prop[0]['orbital_period']} \t\t"
            f"{planet_prop[0]['orbital_period_unit']}"
        )
        print(
            f"Transit Time \t\t{planet_prop[0]['transit_time'] + 0.5} \t\t"
            f"{planet_prop[0]['transit_time_unit']}"
        )
        print(f"Planet Mass Reference: {planet_prop[0]['Mp_ref']}")

    def download_data(self, product=None):
        """
        Downloads data from the specified planet and returns
        the manifest of downloaded products.

        Returns:
            manifest (str): The manifest of downloaded products.
        """
        if product is None:
            product = ["LC"]
        observations = Observations.query_object(self.planet_name, radius="0 deg")
        obs_wanted = (observations["dataproduct_type"] == "timeseries") & (
            observations["obs_collection"] == "TESS"
        )
        print(observations[obs_wanted]["obs_collection", "project", "obs_id"])
        data_products = Observations.get_product_list(observations[obs_wanted])
        products_wanted = Observations.filter_products(
            data_products, productSubGroupDescription=product
        )

        print(products_wanted["productFilename"])
        manifest = Observations.download_products(
            products_wanted, download_dir=self.full_datadir
        )
        print("\nfinished!")
        return manifest

    def de(self, niter=200, npop=30, datadir=None):
        """
        Perform differential evolution optimization.

        Args:
            niter (int): Number of iterations for the optimization.
            Default is 200.
            npop (int): Number of individuals in the population.
            Default is 30.
            datadir (str): Directory path for the data. Default is None.

        Returns:
            None
        """
        zero_epoch = self.zero_epoch
        period = self.period

        self.lpf = TESSLPF(
            self.planet_name,
            datadir,
            tic=self.ticid,
            zero_epoch=zero_epoch.n,
            period=period.n,
            use_pdc=True,
            nsamples=2,
            bldur=0.25,
        )

        ep = epoch(self.lpf.times[0].mean(), self.zero_epoch.n, self.period.n)
        tc = zero_epoch + ep * period

        self.lpf.set_prior(
            "tc", "NP", tc.n, 0.005
        )  # Wide normal prior on the transit center
        self.lpf.set_prior(
            "p", "NP", period.n, period.s
        )  # Wide normal prior on the orbital period
        self.lpf.set_prior("rho", "UP", 0, 1)
        # Uniform prior on the stellar density
        self.lpf.set_prior("k2", "UP", 0.0, 0.2**2)
        # Uniform prior on the area ratio
        self.lpf.set_prior(
            "gp_ln_in", "UP", -2, 1
        )  # Uniform prior on the GP input scale
        self.lpf.optimize_global(niter=niter, npop=npop)

    def plot_original_data(self):
        """
        Plot the original data.

        Returns:
            fig (matplotlib.figure.Figure): The figure object.
            ax (matplotlib.axes.Axes): The axes object.
        """
        timea = self.lpf.timea
        fluxa = self.lpf.ofluxa
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        plt.plot(timea, fluxa)
        return fig, ax

    def fit_single(self, i, niter=100, npop=50, mcmc_repeats=4):
        """
        Fits a single transit light curve.

        Args:
            i (int): Index of the light curve to fit.
            niter (int, optional): Number of iterations for
            global optimization. Defaults to 100.
            npop (int, optional): Population size for global optimization.
            Defaults to 50.
            mcmc_repeats (int, optional): Number of MCMC repeats.
            Defaults to 4.

        Returns:
            SingleFit: The fitted single transit light curve.
        """
        single = SingleFit(
            self.planet_name + str(i) + "th",
            None,
            self.lpf.times[i],
            self.lpf.fluxes[i],
        )
        ep = epoch(single.timea.mean(), self.zero_epoch.n, self.period.n)
        tc = self.zero_epoch + ep * self.period

        single.set_prior(
            "tc", "NP", tc.n, tc.s
        )  # Wide normal prior on the transit center
        single.set_prior(
            "p", "NP", self.period.n, self.period.s
        )  # Wide normal prior on the orbital period
        single.set_prior("rho", "UP", 0, 1)
        # Uniform prior on the stellar density
        single.set_prior("k2", "UP", 0.0, 0.2**2)
        # Uniform prior on the area ratio
        single.optimize_global(niter=niter, npop=npop)
        single.sample_mcmc(2500, thin=25, repeats=mcmc_repeats)

        return single

    def fit_singles(self):
        """
        Fits transit models to individual light curves.

        Args:
            niter (int): Number of iterations for fitting the transit model.
            Default is 100.
            npop (int): Number of populations for the genetic algorithm.
            Default is 50.

        Returns:
            None
        """
        self.fit_single_v = np.vectorize(self.fit_single)
        self.singles = self.fit_single_v(np.arange(len(self.lpf.times)))
        self.post_samples = []
        self.tcs = []
        for single in self.singles:
            df = single.posterior_samples()
            self.post_samples.append(df)
            tc = ufloat(df["tc"].mean(), df["tc"].std())
            self.tcs.append(tc)

        self.epochs = epoch_v(getn_v(self.tcs), self.zero_epoch.n, self.period.n)

    def get_posterior_samples(self):
        """
        Retrieves the posterior samples for each single transit in the dataset.

        Returns:
            None
        """
        self.post_samples = []
        self.tcs = []
        for single in self.singles:
            df = single.posterior_samples()
            self.post_samples.append(df)
            tc = ufloat(df["tc"].mean(), df["tc"].std())
            self.tcs.append(tc)
        self.epochs = epoch_v(getn_v(self.tcs), self.zero_epoch.n, self.period.n)

    def calculate_ttv(self):
        """
        Calculate the Transit Timing Variations (TTV) for the given object.

        Returns:
            None
        """
        day_to_s = 24 * 60 * 60
        tcs = getn_v(self.tcs)
        epochs = self.epochs
        self.ttv_err = (
            gets_v(self.tcs) + self.period.s * (epochs - epochs[0])
        ) * day_to_s

        def f_1(x, a, b):
            return a * x + b

        a, b = curve_fit(f_1, epochs - epochs[0], tcs - tcs[0])[0]

        self.ttv_mcmc_raw = (tcs - tcs[0] - a * (epochs - epochs[0]) + b) * DAY_TO_SEC
        self.ttv_mcmc = self.ttv_mcmc_raw - self.ttv_mcmc_raw.mean()

    def plot_tcs(self, plot_zero_epoch=False):
        """
        Plots the transit centers as a function of epoch.

        Args:
            plot_zero_epoch (bool): If True,
            the zero epoch will be plotted as well.

        Returns:
            fig (matplotlib.figure.Figure): The figure object.
            ax (matplotlib.axes.Axes): The axes object.
        """
        epochs = self.epochs
        tcs = self.tcs
        if plot_zero_epoch is True:
            tcs = np.insert(tcs, 0, self.zero_epoch)
            epochs = np.insert(epochs, 0, 0)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        # plt.plot(epochs,getn_v(self.tcs),'.')
        plt.errorbar(epochs, getn_v(tcs), yerr=gets_v(tcs), fmt="o")
        plt.xlabel("Epoch")
        plt.ylabel(r"$t_c$ (MJD)")

        return fig, ax

    def plot_ttv_re(
        self, plot_zero_epoch=False, set_epoch_zero=False, remove_baseline=True
    ):
        """
        Plot the TTV (transit timing variation) residuals.

        Args:
            plot_zero_epoch (bool): If True, plot the zero epoch.
            set_epoch_zero (bool): If True, set the first epoch to zero.
            remove_baseline (bool): If True, remove the baseline from the plot.

        Returns:
            fig (matplotlib.figure.Figure): The figure object.
            ax (matplotlib.axes.Axes): The axes object.
        """
        epochs = self.epochs
        ttv_mcmc = self.ttv_mcmc
        ttv_raw = self.ttv_mcmc_raw
        ttv_err = self.ttv_err

        if plot_zero_epoch is True:
            ttv_mcmc = np.insert(ttv_mcmc, 0, 0)
            ttv_mcmc_raw = np.insert(ttv_raw, 0, 0)
            ttv_err = np.insert(ttv_err, 0, self.zero_epoch.s)
            epochs = np.insert(epochs, 0, 0)

        if set_epoch_zero is True:
            if plot_zero_epoch is False:
                epochs = epochs - epochs[0]
            else:
                raise ValueError(
                    "Cannot set plot_zero_epoch and \
                        set_epoch_zero = True at the same time"
                )

        if remove_baseline is True:
            ttv = ttv_mcmc
        else:
            ttv = ttv_mcmc_raw

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        plt.errorbar(epochs, ttv, yerr=ttv_err, fmt="o")

        # plt.plot(epochs,getn_v(re)*day_to_s,linestyle='',marker='.',markersize=12)
        plt.xlabel("Epoch")
        plt.ylabel("residual (s)")

        return fig, ax
