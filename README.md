
[![Build Status](https://github.com/BarbourLab/lussac/actions/workflows/testing-dataset.yml/badge.svg)](https://github.com/BarbourLab/lussac/actions/workflows/testing-dataset.yml/badge.svg) [![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://img.shields.io/badge/python-3.10-blue.svg)

# Lussac v1.1

Lussac is an **automated** and **configurable** analysis pipeline for post-processing and merging multiple spike-sorting analyses. The goal is to improve the yield and quality of data from multielectrode extracellular recordings by merging the outputs of different spike-sorting algorithms and/or multiple runs with different parameters.
For more information, check out our [preprint](https://www.biorxiv.org/content/10.1101/2022.02.08.479192v1).


## Installation and usage

Information on how to install and use Lussac can be found on the [wiki page](https://github.com/BarbourLab/lussac/wiki).


## Results

We tested our algorithm using a **synthetic** data set simulating cortical pyramidal cell and interneurone activity ([Jun et al. 2017](https://www.biorxiv.org/content/10.1101/101030v2)) available through [SpikeForest](https://spikeforest.flatironinstitute.org/). Since all spike times of all neurones are known, we could easily compare our algorithm to standard runs of individual spike-sorting algorithms, in terms of recovery and contaminating events.

![Lussac synthetic results](https://github.com/BarbourLab/lussac/blob/main/img/results_synthetic.png?raw=true)
