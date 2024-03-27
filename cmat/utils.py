import numpy as np
import requests
from matplotlib import colors
from pytransit.orbits import epoch
import os

planeturl = "https://exo.mast.stsci.edu/api/v0.1/exoplanets/"
dvurl = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/"
url = planeturl + "/identifiers/"
header = {}


def get_id(planet_name: str):
    myparams = {"name": planet_name}
    url = planeturl + "/identifiers/"
    r = requests.get(url=url, params=myparams, headers=header)
    # print(r.headers.get('content-type'))
    planet_names = r.json()
    ticid = planet_names["tessID"]

    return ticid


def get_prop(planet_name: str, tic: int):
    url = planeturl + planet_name + "/properties/"
    r = requests.get(url=url, headers=header)
    url = planeturl + planet_name + "/properties/"
    r = requests.get(url=url, headers=header)

    return r.json()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


# create a new directory and save data, while also checking if the folder
# exists and asking the user if they want to create a new folder, and if the
# file already exists, asking if the user wants to overwrite it
def save_df_data(dir_path, file_name, df_data):

    # Check if the directory exists:
    if not os.path.exists(dir_path):
        # The directory doesn't exist, so ask the user if they want to create
        # it:
        create_dir = input(
            "The directory doesn't exist. Would you like to create it? (y/n) ")
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
            "The file already exists. Do you want to overwrite it? (y/n) ")
        if overwrite_file.lower() != "y":
            print("Unable to save data.")
            exit()

    # Save some example data to the file:
    df_data.to_csv(file_path, index=False)

    print("Data saved successfully!")


def read_data(name: str):
    with open(name, "r") as file:
        return file


def getn(unfloat):
    return unfloat.n


def gets(unfloat):
    return unfloat.s


getn_v = np.vectorize(getn)
gets_v = np.vectorize(gets)
epoch_v = np.vectorize(epoch)
