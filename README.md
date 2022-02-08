
# Lussac (Beta version)

Lussac is an **automated** and **configurable** analysis pipeline for post-processing and merging multiple spike-sorting analyses. The goal is to improve the yield and quality of data from multielectrode extracellular recordings by merging the outputs of different spike-sorting algorithms and/or multiple runs with different parameters.
For more information, check out our preprint: link to come.

Please note that this project is still in its beta phase.


## Installation and usage

Information on how to install and use can be found in the [wiki page](https://github.com/BarbourLab/lussac/wiki).


## Results

We tested our algorithm using a **synthetic** data set simulating cortical pyramidal cell and interneurone activity ([Jun et al. 2017](https://www.biorxiv.org/content/10.1101/101030v2)) available through [SpikeForest](https://spikeforest.flatironinstitute.org/). Since all spike times of all neurones are known, we could easily compare our algorithm to standard runs of a single spike-sorting algorithms in terms of recovery and contaminating events.

![Lussac synthetic results](https://github.com/BarbourLab/lussac/blob/main/img/results_synthetic.png?raw=true)
