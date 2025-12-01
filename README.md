# COS numerical pipelines and empirical tests

This repository collects research code related to the **COS** (Collapsing
Structure) framework. It focuses on numerical pipelines and empirical tests
used in the COS-NUM and COS-EXP papers, but may also contain additional
exploratory scripts and utilities.

The main goals of this repository are:

- to provide **reference implementations** of the key COS analysis pipelines,
- to document how the **CMB time-arrow**, **SGWB**, and **LSS/DESI** tests are
  actually computed in practice,
- to make it possible to **reproduce the figures and tables** in the COS-NUM
  and COS-EXP articles, given access to the corresponding public survey data.

The repository is designed to remain flexible: over time, new scripts and
modules may be added as the COS program evolves.

---

## Repository contents (core scripts)

The exact layout may change as the project is cleaned up and refactored, but
typical core scripts include:

- `cos_planck_v4_4_0.py` – COS-Planck analysis pipeline. Handles Planck 2018
  CMB maps and masks, Monte Carlo ensembles, HEALPix backends (`healpy`,
  `ducc0`), and COS-specific statistics.

- `cmb_time_arrow_MI_scan_axes.py` – mutual-information–based CMB time-arrow
  pipeline. For a distinguished COS axis and a control set of random axes it
  computes scale-dependent ΔMI(ℓ_max) curves and monotonicity statistics,
  producing JSON outputs that can be used for further analysis.

- `cos_cmb_timearrow_bayes.py` – Bayesian post-processing of the MI
  time-arrow results. Fits simple parametric models (constant, linear,
  COS-specific linear, quadratic) to the averaged ΔMI(ℓ_max) curve using
  nested sampling (`dynesty`), and quantifies the strength of the preference
  for a non-constant trend. Also evaluates the global significance of the
  COS-axis monotonicity compared to ΛCDM random axes.

- `cos_sgwb_cosplusastro.py` – numerical pipeline for stochastic
  gravitational-wave background (SGWB) analyses, combining astrophysical and
  COS components. Uses standard GW tools (`pygwb`, `gwpy`, `bilby`, `dynesty`)
  to compute evidences, Bayes factors and posterior distributions for SGWB
  models.

- `cos_desi_tests.py` – scripts to test COS predictions against large-scale
  structure data (e.g. DESI-like surveys). Includes topological or
  morphological statistics (e.g. Minkowski functionals, filamentarity-related
  quantities) and basic consistency checks with COS forecasts.

Additional helper modules, run scripts (`.sh`, `.yaml`) and small JSON/NPY
summary files may appear as needed. Large raw survey data (Planck, DESI, GW
catalogs, etc.) are **not** stored in this repository.

---

## Installation / environment

There are many possible ways to install the required Python packages. Below
are two example workflows that have been tested with the COS-related scripts.

In all cases it is strongly recommended to use an **isolated environment**
(virtualenv, conda/micromamba) to avoid conflicts with system packages.

### Option A: Python virtual environment (Linux / WSL + pip)

On Windows, a convenient setup is to use **WSL2** with an Ubuntu distribution:

1. Install WSL and Ubuntu:

   `wsl --install -d Ubuntu`

2. Inside the Linux/WSL shell install system Python and tools:

   - `sudo add-apt-repository ppa:deadsnakes/ppa`
   - `sudo apt update`
   - `sudo apt install -y python3.11 python3.11-venv python3-pip gfortran`

3. Create and activate a virtual environment:

   - `python3.11 -m venv ~/cmbenv`
   - `source ~/cmbenv/bin/activate`

4. Upgrade pip and install the core Python dependencies:

   - `pip install -U pip wheel`
   - `pip install numpy scipy matplotlib astropy tqdm ducc0 healpy dynesty bilby pygwb gwpy`

Once the environment is active (`source ~/cmbenv/bin/activate`), the COS
scripts can be run, for example:

- `python cos_planck_v4_4_0.py`

### Option B: micromamba / conda-forge environment

Alternatively, you can use a conda-style environment with **micromamba**:

1. Download micromamba (Linux):

   - `cd ~`
   - `curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba`

2. Initialize micromamba:

   - `export MAMBA_ROOT_PREFIX="$HOME/micromamba"`
   - `./bin/micromamba shell init -s bash`
   - `exec bash`   (restart the shell so that \`micromamba\` is available)

3. Create and activate a `cmb` environment:

   - `micromamba create -y -n cmb -c conda-forge python=3.11 numpy scipy matplotlib astropy tqdm ducc0 healpy dynesty bilby pygwb gwpy`
   - `micromamba activate cmb`

After activation, the scripts can be run as usual, e.g.:

- `python cos_cmb_timearrow_bayes.py`

You can adapt these examples to your own platform (native Linux, macOS, etc.)
as long as the listed dependencies are installed.

---

## Data

The COS pipelines rely on **public cosmological data sets**, which are not
bundled with this repository for reasons of size and licensing. Instead,
please obtain them directly from their official archives, for example:

- Planck 2018 CMB maps and masks (SMICA, NILC, SEVEM, Commander, common
  masks) – Planck Legacy Archive / ESA archive (insert official link here).

- Large-scale structure / DESI data products – DESI data release (insert
  official link here).

- Stochastic gravitational-wave background (SGWB) and strain data –
  LIGO/Virgo/KAGRA open data archives (insert official link here).

The COS-NUM and COS-EXP papers, together with their appendices and
supplementary materials, specify:

- which exact maps and masks were used (file names, Nside, smoothing, etc.),
- which survey releases and cuts were applied,
- what random seed conventions and numerical parameters were adopted.

The JSON/NPY summary files and configuration examples in this repository are
intended to be sufficient to reproduce the **published figures and tables**
once the required public data sets have been downloaded.

---

## Reproducibility and versioning

To support scientific reproducibility, the COS code follows these principles:

- Scripts contain comments and (where practical) version strings or references
  to specific COS-NUM / COS-EXP versions.
- Run configurations (`.sh`, `.yaml`, command-line examples) explicitly list
  the input files, masks, seeds and numerical parameters used in published
  runs.
- The COS-EXP and COS-NUM papers cite this repository (and, where applicable,
  a Zenodo DOI snapshot). Future updates, bugfixes and refactorings are
  recorded via git history and/or a changelog.

If you use this code in your own work, please cite the relevant COS papers
and, if appropriate, the archived DOI of this repository.

---

## Disclaimer

This repository is **research code**. While care has been taken to make the
pipelines transparent and reproducible, the scripts are not guaranteed to be
production-grade software. Interfaces, file names and directory structure may
change as the COS program evolves.

Bug reports, questions and suggestions are welcome via GitHub issues.
