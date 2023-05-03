[![Python version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://img.shields.io/badge/python-3.10-blue.svg)
[![Documentation Status](https://readthedocs.org/projects/lussac/badge/?version=latest)](http://lussac.readthedocs.io/)
[![Build status](https://github.com/BarbourLab/lussac/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/BarbourLab/lussac/actions/workflows/unit-tests.yml)
[![Coverage report](https://codecov.io/gh/barbourlab/lussac/graphs/badge.svg)](https://app.codecov.io/github/barbourlab/lussac)

# Lussac 2.0

:warning: Lussac 2.0 is still in development and is not yet operational! :warning:

Lussac is an **automated** and **configurable** analysis pipeline for post-processing and/or merging multiple spike-sorting analyses. The goal is to improve the **yield** and **quality** of data from multielectrode extracellular recordings by comparing the outputs of different spike-sorting algorithms and/or multiple runs with different parameters. For more information, check out our [preprint](https://www.biorxiv.org/content/10.1101/2022.02.08.479192v1).


## Installation

```bash
# Download Lussac in any directory you want.
git clone https://github.com/BarbourLab/lussac.git
cd lussac
git checkout v2.0

# OPTIONAL: Use a conda environment.
conda create -n lussac python=3.11  # Must be >= 3.10
conda activate lussac

# Install Lussac.
pip install -e .
# Or for developers: pip install -e .[dev]

# If you want to check whether the installation was successful (optional)
# (this may take a while as it will download some testing datasets).
pytest
```

You will also need the latest development version of [SpikeInterface](https://github.com/SpikeInterface/spikeinterface)


## Documentation

You can find the documentation [here](https://lussac.readthedocs.io/).


## Migration from Lussac1

Lussac2 is not backwards-compatible with Lussac1.  We advise you to make a new conda environment, and to remake your `params.json` file (which is also not backwards-compatible).
