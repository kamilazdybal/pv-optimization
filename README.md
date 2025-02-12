# 📄 Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks

This repository contains code, datasets, and results from the paper:

> Kamila Zdybał, James C. Sutherland, Alessandro Parente - *Optimizing progress variables for ammonia/hydrogen combustion using encoding-decoding networks*, 2025.

<p align="center">
  <img src="https://github.com/kamilazdybal/pv-optimization/raw/main/figures/ED-for-PV-optimization.png" width="500">
</p>

## Data and results files

Data and results files will be shared separately via GoogleDrive as they take over 5GB of space.

- Script for loading data [`ammonia-Stagni-load-data.py`](scripts/ammonia-Stagni-load-data.py)

## Requirements

We have used `Python==3.10.13` and the following versions of all libraries:

```bash
pip install numpy==1.26.2
pip install pandas==2.1.3
pip install scipy==1.11.4
pip install scikit-learn==1.3.2
pip install tensorflow==2.15.0
pip install keras==2.15.0
```

You will also need our library [`PCAfold==2.2.0`](https://pcafold.readthedocs.io/en/latest/index.html).

Other requirements are:

```bash
pip install matplotlib
pip install plotly
pip install cmcrameri
```

## Python scripts

Python scripts stored in `scripts/` allow you to train encoders-decoders and assess the optimized PVs.
The scripts will produce results saved as `.csv` and `.h5` files that you can later post-process with dedicated
Jupyter notebooks stored in `jupyter-notebooks/`.

### The order of executing scripts

First, run the PV optimization with [`RUN-PV-optimization.py`](scripts/RUN-PV-optimization.py) with desired parameters. 
Once you have the results files, you can run quantitative assessment of PVs with [`RUN-VarianceData.py`](scripts/RUN-VarianceData.py). 
Both those scripts load the appropriate data under the hood using [`ammonia-Stagni-load-data.py`](scripts/ammonia-Stagni-load-data.py).

You have a lot of flexibility in setting different ANN hyper-parameters in those two scripts using the `argparse` Python library.
If you're new to `argparse`, check out my short video tutorials:

- [Intro to argparse](https://youtu.be/ONCv_ql2xpE)
- [Setting booleans with argparse](https://youtu.be/8gfFteE6jz0)

#### Optimizing PVs

- Master script for running PV optimization [`RUN-PV-optimization.py`](scripts/RUN-PV-optimization.py)

The above script uses one of the following under the hood:

- QoI-aware encoder-decoder for the $(f, PV)$ optimization [`QoI-aware-ED-f-PV.py`](scripts/QoI-aware-ED-f-PV.py)
- QoI-aware encoder-decoder for the $(f, PV, \gamma)$ optimization [`QoI-aware-ED-f-PV-h.py`](scripts/QoI-aware-ED-f-PV-h.py)

depending on which `--parameterization` you selected.

#### Quantitative assessment of PVs

- Master script for running PV optimization [`RUN-VarianceData.py`](scripts/RUN-VarianceData.py)

The above script uses one of the following under the hood:

- Assessment of $(f, PV)$ parameterizations [`VarianceData-f-PV.py`](scripts/VarianceData-f-PV.py)
- Assessment of $(f, PV, \gamma)$ parameterizations [`VarianceData-f-PV-h.py`](scripts/VarianceData-f-PV-h.py)

depending on which `--parameterization` you selected.

### Running Python jobs

This is a minimal example for running a Python script with all hyper-parameters set as per §2.2 in the paper:

```bash
python RUN-PV-optimization.py --parameterization 'f-PV' --data_type 'SLF' --data_tag 'NH3-H2-air-25perc' --random_seeds_tuple 0 20 --target_variables_indices 0 1 3 5 6 9
```

Alternatively, you can change various parameters (kernel initializer, learning rate, *etc*.) using the appropriate argument:

```bash
python RUN-PV-optimization.py --initializer 'GlorotUniform' --init_lr 0.001 --parameterization 'f-PV' --data_type 'SLF' --data_tag 'NH3-H2-air-25perc' --random_seeds_tuple 0 20 --target_variables_indices 0 1 3 5 6 9
```

If you'd like to remove pure stream components from the PV definition (**non-trainable pure streams** preprocessing as discussed in §3.4. in the paper) use the flag:

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

Results generated with the Python scripts described above can be post-processed and visualized 
in dedicated Jupyter notebooks stored in `jupyter-notebooks/`.
You can access the appropriate notebook below:

### Reproducing Fig. 2 and Fig. 3 - Quantitative assessment of the optimized PVs

→ This [Jupyter notebook](jupyter-notebooks/Figure-02-03-Quantitative-assessment-of-the-optimized-PVs.ipynb) can be used to reproduce Fig. 2 and Fig. 3.

### Reproducing Fig. 4 - Physical insight into the optimized PVs

→ This [Jupyter notebook](jupyter-notebooks/Figure-04-Physical-insight-into-the-optimized-PVs.ipynb) can be used to reproduce Fig. 4

### Reproducing Fig. 5, Fig. 6, and Fig. 7 - Towards flamelet-like model adaptivity

→ This [Jupyter notebook](jupyter-notebooks/Figure-05-06-07-Towards-flamelet-like-model-adaptivity.ipynb) can be used to reproduce Fig. 5, Fig. 6, and Fig. 7

### Reproducing Fig. 8, Fig. 9, and Fig. 10 - The effect of trainable vs. non-trainable pure streams

→ This [Jupyter notebook](jupyter-notebooks/Figure-08-09-10-Non-trainable-pure-streams.ipynb) can be used to reproduce Fig. 8, Fig. 9, and Fig. 10

### Reproducing supplementary Figs. S37-S38 - The effect of scaling encoder inputs prior to training

→ This [Jupyter notebook](jupyter-notebooks/SUPPLEMENT-Effect-of-scaling-encoder-inputs.ipynb) can be used to reproduce supplementary Figs. S37-S38.
