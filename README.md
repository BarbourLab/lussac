[![Python version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://img.shields.io/badge/python-3.10-blue.svg)
[![Documentation Status](https://readthedocs.org/projects/lussac/badge/?version=latest)](http://lussac.readthedocs.io/)
[![Build status](https://github.com/BarbourLab/lussac/actions/workflows/unit-tests-linux.yml/badge.svg)](https://github.com/BarbourLab/lussac/actions/workflows/unit-tests.yml)
[![Coverage report](https://codecov.io/gh/barbourlab/lussac/graphs/badge.svg)](https://app.codecov.io/github/barbourlab/lussac)

# Lussac 2.0

:warning: Lussac 2.0 is in beta! :warning:

You can use the version 1 with `git checkout v1`

Lussac is an **automated** and **configurable** analysis pipeline for post-processing and/or merging multiple spike-sorting analyses. The goal is to improve the **yield** and **quality** of data from multielectrode extracellular recordings by comparing the outputs of different spike-sorting algorithms and/or multiple runs with different parameters. For more information, check out our [preprint](https://www.biorxiv.org/content/10.1101/2022.02.08.479192v1).


## Installation

You can install the latest release version of Lussac:

```bash
# OPTIONAL: Use a conda environment.
conda create -n lussac python=3.11  # Must be >= 3.10
conda activate lussac

pip install lussac
# pip install --upgrade lussac  # To upgrade in case a new version is released.
```

Or if you prefer downloading the latest developmental version:

```bash
# Download Lussac in any directory you want.
git clone https://github.com/BarbourLab/lussac.git --branch dev
cd lussac

# OPTIONAL: Use a conda environment.
conda create -n lussac python=3.11  # Must be >= 3.10
conda activate lussac

# Install Lussac.
pip install -e .[dev]

# To upgrade Lussac.
git pull

# If you want to check whether the installation was successful (optional)
pytest
```


## Documentation

You can find the documentation [here](https://lussac.readthedocs.io/).


## Migration from Lussac1

Lussac2 is not backwards-compatible with Lussac1.  We advise you to make a new conda environment, and to remake your `params.json` file (which is also not backwards-compatible).