# 📄 Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks

This repository contains code, datasets, and results from the paper:

> Kamila Zdybał, James C. Sutherland, Alessandro Parente - *Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks*, 2024.

<p align="center">
  <img src="https://github.com/kamilazdybal/pv-optimization/raw/main/figures/ED-for-PV-optimization.png" width="500">
</p>

## Data

- Script for loading data [`ammonia-Stagni-load-data.py`](code/ammonia-Stagni-load-data.py)

## Code

### The order of executing scripts

First, run the PV optimization with [`RUN-PV-optimization.py`](code/RUN-PV-optimization.py) with appropriate parameters. Once you have the results files, you can
run quantitative assessment of PVs with [`RUN-VarianceData.py`](code/RUN-VarianceData.py).

You have a lot of flexibility in setting different ANN hyper-parameters in those two scripts using the `argparse` Python library.
If you're new to `argparse`, check out my short video tutorials:

- [Intro to argparse](https://youtu.be/ONCv_ql2xpE)
- [Setting booleans with argparse](https://youtu.be/8gfFteE6jz0)

### Optimizing PVs

- Master script for running PV optimization [`RUN-PV-optimization.py`](code/RUN-PV-optimization.py)

The above script uses one of the following under the hood:

- QoI-aware encoder-decoder for the $(f, PV)$ optimization [`QoI-aware-ED-f-PV.py`](code/QoI-aware-ED-f-PV.py)
- QoI-aware encoder-decoder for the $(f, PV, \gamma)$ optimization [`QoI-aware-ED-f-PV-h.py`](code/QoI-aware-ED-f-PV-h.py)

depending on which `--parameterization` you selected.

### Quantitative assessment of PVs

- Master script for running PV optimization [`RUN-VarianceData.py`](code/RUN-VarianceData.py)

The above script uses one of the following under the hood:

- Assessment of $(f, PV)$ parameterizations [`VarianceData-f-PV.py`](code/VarianceData-f-PV.py)
- Assessment of $(f, PV, \gamma)$ parameterizations [`VarianceData-f-PV-h.py`](code/VarianceData-f-PV-h.py)

depending on which `--parameterization` you selected.

### Running Python jobs

This is a minimal example for running a Python script with all hyper-parameters set as per section 2.2 in the paper:

```bash
python RUN-PV-optimization.py --parameterization 'f-PV' --data_type 'SLF' --data_tag 'NH3-H2-air-25perc' --random_seeds_tuple 0 20 --target_variables_indices 0 1 3 5 6 9
```

Alternatively, you can change various parameters (kernel initializer, learning rate, *etc*.) using the appropriate argument:

```bash
python RUN-PV-optimization.py --initializer 'GlorotUniform' --init_lr 0.001 --parameterization 'f-PV' --data_type 'SLF' --data_tag 'NH3-H2-air-25perc' --random_seeds_tuple 0 20 --target_variables_indices 0 1 3 5 6 9
```

If you'd like to remove pure stream components from the PV definition (**non-trainable pure streams** preprocessing as discussed in section 3.4. in the paper) use the flag:

```bash
--no-pure_streams
```

as an extra argument.

To run $(f, PV)$ optimization, use:

```bash
--parameterization 'f-PV'
```

To run $(f, PV, \gamma)$ optimization, use:

```bash
--parameterization 'f-PV-h'
```

> **Note:** Logging with [Weights & Biases](https://wandb.ai/site) is also possible in the scripts above.

## Jupyter notebooks

All results are post-processed and visualized in dedicated Jupyter notebooks. You can access the appropriate notebook below:

### Reproducing Figs. 2-3

→ This [Jupyter notebook](jupyter-notebooks/Figure-02-03-Quantitative-assessment-of-the-optimized-PVs.ipynb) can be used to reproduce Figs. 2-3.

### Reproducing Fig. 4 and Fig. 10

→ This [Jupyter notebook](jupyter-notebooks/Figure-04-Physical-insight-into-the-optimized-PVs.ipynb) can be used to reproduce Fig. 4 and Fig. 10.

### Reproducing supplementary Figs. S37-S38

→ This [Jupyter notebook](jupyter-notebooks/SUPPLEMENT-Effect-of-scaling-encoder-inputs.ipynb) can be used to reproduce supplementary Figs. S37-S38.
