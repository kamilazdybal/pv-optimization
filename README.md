# 📄 Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks

This repository contains code, datasets, and results from the paper:

> Kamila Zdybał, James C. Sutherland, Alessandro Parente - *Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks*, 2024.

<p align="center">
  <img src="https://github.com/kamilazdybal/pv-optimization/raw/main/figures/ED-for-PV-optimization.png" width="300">
</p>

## Data

- Script for loading data [`ammonia-Stagni-load-data.py`](code/ammonia-Stagni-load-data.py)

## Code

> **Note:** Logging with [Weights & Biases](https://wandb.ai/site) is possible in the scripts below. 

### Optimizing PVs

- QoI-aware encoder-decoder for the $(f, PV)$ optimization [`QoI-aware-ED-f-PV.py`](code/QoI-aware-ED-f-PV.py)
- QoI-aware encoder-decoder for the $(f, PV, \gamma)$ optimization [`QoI-aware-ED-f-PV-h.py`](code/QoI-aware-ED-f-PV-h.py)

- Master script for running PV optimization [`RUN-PV-optimization.py`](code/RUN-PV-optimization.py)

### Quantiative assessment of PVs

- Assessment of $(f, PV)$ parameterizations [`VarianceData-f-PV.py`](code/VarianceData-f-PV.py)
- Assessment of $(f, PV, \gamma)$ parameterizations [`VarianceData-f-PV-h.py`](code/VarianceData-f-PV-h.py)

- Master script for running PV optimization [`RUN-VarianceData.py`](code/RUN-VarianceData.py)

### Running Python jobs

This is a minimal example for running a Python script with all hyper-parameters set as per section 2.2 in the paper:

```bash
python RUN-PV-optimization.py --parameterization 'f-PV' --data_type 'SLF' --data_tag 'NH3-H2-air-25perc' --random_seeds_tuple 0 20 --target_variables_indices 0 1 3 5 6 9
```

Alternatively, you can change various parameters (kernel initializer, learning rate, etc.) using the appropriate argument:

```bash
python RUN-PV-optimization.py --no-pure-streams --initializer 'GlorotUniform' --init_lr 0.001 --parameterization 'f-PV' --data_type 'SLF' --data_tag 'NH3-H2-air-25perc' --random_seeds_tuple 0 20 --target_variables_indices 0 1 3 5 6 9
```

## Jupyter notebooks

### Reproducing Figs. 2-3

→ This [Jupyter notebook](jupyter-notebooks/Figure-02-03-Quantitative-assessment-of-the-optimized-PVs.ipynb) can be used to reproduce Figs. 2-3.