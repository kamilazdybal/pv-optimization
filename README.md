# 📄 Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks

This repository contains code, datasets, and results from the paper:

> K. Zdybał, James C. Sutherland, Alessandro Parente - *Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks*, 2024.

## Data

- [Script for loading data](code/ammonia-Stagni-load-data.py)

## Code

### Optimizing PVs

- [QoI-aware encoder-decoder for the $(f, PV)$ optimization](code/QoI-aware-ED-f-PV.py)
- [QoI-aware encoder-decoder for the $(f, PV, \gamma)$ optimization](code/QoI-aware-ED-f-PV-h.py)

- [Master script for running PV optimization](code/RUN-PV-optimization.py)

### Quantiative assessment of PVs

- [Assessment of $(f, PV)$ parameterizations](code/VarianceData-f-PV.py)
- [Assessment of $(f, PV, \gamma)$ parameterizations](code/VarianceData-f-PV-h.py)

- [Master script for running PV optimization](code/RUN-VarianceData.py)

## Jupyter notebooks