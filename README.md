[![Python version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://img.shields.io/badge/python-3.10-blue.svg)
[![Build status](https://github.com/BarbourLab/lussac/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/BarbourLab/lussac/actions/workflows/unit-tests.yml)
[![Coverage report](https://codecov.io/gh/barbourlab/lussac/branch/v2.0/graphs/badge.svg)](https://codecov.io/github/barbourlab/lussac/branch/v2.0)

# Lussac 2.0

Lussac 2.0 is still in development and is not yet operational.


## Installation

```bash
# Download Lussac in any directory you want.
git clone https://github.com/BarbourLab/lussac.git
cd lussac

# OPTIONAL: Use a conda environment.
conda create -n lussac python=3.10
conda activate lussac

# Install Lussac.
pip install -r requirements.txt
python setup.py install
```

If you want to check whether the installation was successful, you can run the tests (this may take a while as it will download some testing datasets):

```bash
python -m pytest
```
