#################################################################################################################################

import argparse
import numpy as np
import pandas as pd
import time
import pickle
import os
import h5py
from PCAfold import preprocess
from PCAfold import analysis
from PCAfold import __version__ as PCAfold_version

#################################################################################################################################
## Argument parser
#################################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--parameterization',   type=str,  default='f-PV', metavar='PARAM', help='Physics-based parameterization, can be f-PV, f-PV-h, or PV1-PV2')
parser.add_argument('--case_name',          type=str,  default='', metavar='CASENAME', help='Case name to be added to all output filenames')
parser.add_argument('--data_tag',           type=str,  default='NH3-H2-air-25perc', metavar='DATATAG', help='Data tag')
parser.add_argument('--data_type',          type=str,  default='SLF', metavar='DATATAG', help='Data type, can be SLF, STLF, AI, FPF')
parser.add_argument('--target_variables_indices',          type=int,  default=[0,2,3,4,5,6,7,9], nargs="+", metavar='TARGETS', help='Indices for the target variables')
parser.add_argument('--pure_streams',       type=bool, default=True, action=argparse.BooleanOptionalAction, metavar='PURESTREAMS', help='Should pure streams components be within the encoder inputs')
parser.add_argument('--n_decoder_arch',     type=int,  default=[0,10,10], nargs="+", metavar='NDECODER', help='Number of added neurons in each decoding layer beyond the n_decoder_outputs that will already be there')
parser.add_argument('--n_epochs',           type=int,  default=500000, metavar='NEPOCHS', help='Number of epochs')
parser.add_argument('--init_lr',            type=float,default=0.01, metavar='LR', help='Initial learning rate')
parser.add_argument('--alpha_lr',           type=float,default=0.001, metavar='ALPHALR', help='Alpha parameter to compute the final learning rate = alpha * initial learning rate')
parser.add_argument('--n_decay_steps',      type=int,  default=300000, metavar='NDECAYSTEPS', help='Number of decay steps for the CosineDecay learning rate scheduler')
parser.add_argument('--rmsprop_rho',        type=float,default=0.9, metavar='RHO', help='Discounting factor for the old gradients.')
parser.add_argument('--rmsprop_momentum',   type=float,default=0.5, metavar='MOMENTUM', help='Tracking momentum.')
parser.add_argument('--rmsprop_centered',   type=bool, default=False, action=argparse.BooleanOptionalAction, metavar='CENTERED', help='Moment centering.')
parser.add_argument('--validation_perc',    type=int,  default=10, metavar='VALPERC', help='Validation percentage')
parser.add_argument('--scaling',            type=str,  default='none', metavar='SCALING', help='Scaling for the encoder inputs')
parser.add_argument('--initializer',        type=str,  default='RandomNormal', metavar='INITIALIZER', help='Initializer for the weights')
parser.add_argument('--random_seeds_tuple', type=int,  default=[0,10], nargs="+", metavar='SEEDS', help='Min and max random seed')
parser.add_argument('--min_bandwidth',      type=float,  default=-6, metavar='MINBANDWIDTH', help='Log10 of the minimum bandwidth to use for VarianceData computation')
parser.add_argument('--max_bandwidth',      type=float,  default=2, metavar='MAXBANDWIDTH', help='Log10 of the maximum bandwidth to use for VarianceData computation')
parser.add_argument('--n_bandwidths',       type=int,  default=200, metavar='NBANDWIDTHS', help='Number of bandwidths to use for VarianceData computation')
parser.add_argument('--vd_targets',         type=int,  default=[0,2,3,4,5,6,7,9], nargs="+", metavar='VDTARGETS', help='State-space variables for VarianceData computation')

args = parser.parse_args()

print(args)

# Populate values:
parameterization = vars(args).get('parameterization')
case_name = vars(args).get('case_name')
data_tag = vars(args).get('data_tag')
data_type = vars(args).get('data_type')
target_variables_indices = vars(args).get('target_variables_indices')
pure_streams = vars(args).get('pure_streams')
n_decoder_1, n_decoder_2, n_decoder_3 = tuple(vars(args).get('n_decoder_arch'))
n_epochs = vars(args).get('n_epochs')
init_lr = vars(args).get('init_lr')
alpha_lr = vars(args).get('alpha_lr')
n_decay_steps = vars(args).get('n_decay_steps')
rmsprop_rho = vars(args).get('rmsprop_rho')
rmsprop_momentum = vars(args).get('rmsprop_momentum')
rmsprop_centered = vars(args).get('rmsprop_centered')
validation_perc = vars(args).get('validation_perc')
scaling = vars(args).get('scaling')
initializer = vars(args).get('initializer')
min_random_seed, max_random_seed = tuple(vars(args).get('random_seeds_tuple'))
min_bandwidth = vars(args).get('min_bandwidth')
max_bandwidth = vars(args).get('max_bandwidth')
n_bandwidths = vars(args).get('n_bandwidths')
VD_target_variables_indices = vars(args).get('vd_targets')

if rmsprop_centered:
    momcent = 'yes'
else:
    momcent = 'no'

#################################################################################################################################
## Import data
#################################################################################################################################

exec(open('ammonia-Stagni-load-data.py').read())

#################################################################################################################################
## Case settings
#################################################################################################################################

# Master random seed for anything that is not ANN-related:
master_random_seed = 100

# List of random seeds to loop over:
random_seeds_list = [i for i in range(min_random_seed, max_random_seed)]

# Badwidth values for computing VarianceData:
bandwidth_values = np.logspace(min_bandwidth, max_bandwidth, n_bandwidths)

# Case run name that is added to the output files:
if case_name == '':
    case_run_name = pure_streams_prefix + '-' + data_type + '-' + data_tag + '-target-' + '-'.join(target_variables_names) + '-' + initializer + '-e-' + str(n_epochs) + '-CD-' + str(n_decay_steps) + '-lr-' + str(init_lr) + '-alr-' + str(alpha_lr) + '-scale-' + scaling + '-darch-' + str(n_decoder_1) + '-' + str(n_decoder_2) + '-' + str(n_decoder_3) + '-rho-' + str(rmsprop_rho) + '-mom-' + str(rmsprop_momentum) + '-momcent-' + momcent
else:
    case_run_name = case_name + '-' + pure_streams_prefix + '-' + data_type + '-' + data_tag + '-target-' + '-'.join(target_variables_names) + '-' + initializer + '-e-' + str(n_epochs) + '-CD-' + str(n_decay_steps) + '-lr-' + str(init_lr) + '-alr-' + str(alpha_lr) + '-scale-' + scaling + '-darch-' + str(n_decoder_1) + '-' + str(n_decoder_2) + '-' + str(n_decoder_3) + '-rho-' + str(rmsprop_rho) + '-mom-' + str(rmsprop_momentum) + '-momcent-' + momcent

#################################################################################################################################
## Pre-process data to allow for warm-start
#################################################################################################################################

if scaling == 'none':

    Yi_CS = state_space[:,1::]
    Yi_sources_CS = state_space_sources[:,1::]

else:

    (Yi_CS, centers, scales) = preprocess.center_scale(state_space[:,1::], nocenter=True, scaling=scaling)
    Yi_sources_CS = state_space_sources[:,1::]/scales

#################################################################################################################################
## Compute VarianceData
#################################################################################################################################

exec(open('VarianceData-' + parameterization + '.py').read())

#################################################################################################################################