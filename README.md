# CMAT (Companion MAss from TTV modeling)
[![DOI](https://zenodo.org/badge/777723832.svg)](https://zenodo.org/doi/10.5281/zenodo.13739646)

CMAT is a fast and efficient Python tool to constrain the upper mass of a hidden companion in planetary systems using Transit Timing Variations (TTV) data. This package allows users to quickly derive mass constraints using a minimal computational setup.

## File Structure

### 1. `cmat/`
   - **Content**: The main program source code.
   - **Description**: This folder contains the core Python scripts responsible for reading photometric data, processing it, running the mass estimation algorithms, and generating results. The code is modular and can be adapted to different datasets and user needs.

### 2. `data/`
   - **Content**: Example photometric datasets from TESS.
   - **File Format**: `.fits`, `.csv`
   - **Description**: These are sample data files demonstrating the format required for transit data input. The files contain time, flux and associated uncertainties, as well as additional orbital or planetary information.

### 3. `example.ipynb`
   - **Content**: Jupyter Notebook example.
   - **Description**: This interactive notebook walks users through the steps of using CMAT. It demonstrates how to load the provided example data, configure analysis parameters, run the CMAT code, and visualize the results. It serves as a step-by-step guide to get started.

### 4. `requirements.txt`
   - **Content**: Python dependencies.
   - **Description**: This file lists all the Python packages required to run CMAT. Users can easily set up their environment by installing these dependencies using pip.

   ```bash
   pip install -r requirements.txt
   ```

## Installation

CMAT can be installed using pip:

```bash
pip install CMAT-astro
```

This will install the necessary components, allowing you to start analyzing TTV data.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/troyzx/CMAT.git
   cd CMAT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the example Jupyter Notebook:
   ```bash
   jupyter notebook example.ipynb
   ```

The notebook will guide you through the process of using CMAT, including data loading, parameter configuration, and results interpretation.

## Example Data

Example datasets are provided in the `data/` folder to demonstrate the expected input format and analysis process. You can replace these datasets with your own photometric data in `.fits` or TTV data in `.csv` format.

## Key Features

- **Fast Mass Estimation**: CMAT is optimized for fast calculations to derive mass constraints based on TTV data.
- **Extensible**: The source code is modular and can be modified to fit specific research needs.
- **Interactive**: The Jupyter notebook provides a hands-on example to guide users through using the tool.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
