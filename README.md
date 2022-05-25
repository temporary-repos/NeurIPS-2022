# Few-Shot Forecasting for High-Dimensional Time Sequences
<p>This repository contains the codebase used in the associated submission for NeurIPS 2022.</p>

## Package Requirements

* Python >= 3.8
* PyTorch >= 1.7
* ipdb
* matplotlib
* numpy
* scipy
* Pillow
* pymunk
* pygame
* torchdiffeq


## All Datasets
Dataloader objects for both datasets are available in ```data_loaders```. 
More information on how the datasets were generated and used is available in the Appendix of Supplementary Material.

1. Bouncing ball data: please refer to the code in ./data/box_generators.
2. Pendulum/Mass spring data: please refer to [this repo](https://github.com/deepmind/dm_hamiltonian_dynamics_suite).

## Baselines
Included are two baselines DKF and DVBF. DKF is included as an available transitionfunction within the ```model/``` folder while DVBF has its own folder structure ```baseline_dvbf```. DVBF was modified from [this repo](https://github.com/Jgmorton/vbf) and more information is included in a README in its folder.

## Configs
We included configuration riles to run DKF, the non-meta Baselines, and the Meta-Model under ```config```. 
Hyperparameters used for each model in each experiment are available in the Appendix.

An example command to run looks like:
```python3 main.py --config meta01```
