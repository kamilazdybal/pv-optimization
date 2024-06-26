import numpy as np
import pandas as pd
import time
import random
import csv
import copy as cp
import heapq
import h5py
import pickle
import cmcrameri.cm as cmc
from scipy.stats import norm
import os

from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold import reconstruction
from PCAfold import utilities

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib import colormaps

import plotly as plty
from plotly import express as px
from plotly import graph_objects as go

from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from  matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

from sklearn import __version__ as sklearn_version
from scipy import __version__ as scipy_version
from PCAfold import __version__ as PCAfold_version
from platform import python_version

print('Python==' + python_version())
print()
print('numpy==' + np.__version__)
print('pandas==' + pd.__version__)
print('scipy==' + scipy_version)
print('scikit-learn==' + sklearn_version)
print('PCAfold==' + PCAfold_version)

cmap = cmc.lajolla

def plot_3d(x, y, z, color, cmap='plasma', s=2, xlabel='x', ylabel='y', zlabel='z'):

    fig = go.Figure(data=[go.Scatter3d(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        mode='markers',
        marker=dict(size=s,
                    color=color.ravel(),
                    colorscale=cmap,
                    opacity=1,
                    colorbar=dict(thickness=20)))])
    
    fig.update_layout(autosize=False,
                      width=1000, height=600,
                      margin=dict(l=65, r=50, b=65, t=90),
                      scene = dict(xaxis_title=xlabel,
                                   yaxis_title=ylabel,
                                   zaxis_title=zlabel))
    
    fig.show()