# 📄 Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks

This repository contains code, datasets, and results from the paper:

> K. Zdybał, James C. Sutherland, Alessandro Parente - *Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks*, 2024.

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

## Jupyter notebooks

### Reproducing Figs.2-3

→ This [Jupyter notebook](Figure-02-03-Quantitative-assessment-of-the-optimized-PVs.ipynb) can be used to reproduce Figs.2-3.