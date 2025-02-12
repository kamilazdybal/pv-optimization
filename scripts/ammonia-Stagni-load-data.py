#################################################################################################################################
## Load combustion data generated using the ammonia chemical mechanism by Stagni et al.:
# 
# Stagni, Alessandro, et al. "Low-and intermediate-temperature ammonia/hydrogen oxidation in a flow reactor: 
# Experiments and a wide-range kinetic modeling." Chemical Engineering Journal 471 (2023): 144577.
#################################################################################################################################

if data_type == 'SLF':
    file_prefix = '../data/SLF-'
elif data_type == 'STLF':
    file_prefix = '../data/STLF-'
elif data_type == 'NA-SLF':
    file_prefix = '../data/non-adiabatic-SLF-'
elif data_type == 'AI':
    file_prefix = '../data/isochoric-adiabatic-closed-HR-'
    time_vector = pd.read_csv('../data/isochoric-adiabatic-closed-HR-' + data_tag + '-time.csv', sep = ',', header=None).to_numpy()
elif data_type == 'FPF':
    file_prefix = '../data/freely-propagating-flame-'
    grid = pd.read_csv('../data/freely-propagating-flame-' + data_tag + '-grid.csv', sep = ',', header=None).to_numpy()

species_to_remove_list = ['N2', 'AR', 'HE']

state_space = pd.read_csv(file_prefix + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()
state_space_sources = pd.read_csv(file_prefix + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()
state_space_names = pd.read_csv(file_prefix + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()
mf = pd.read_csv(file_prefix + data_tag + '-mixture-fraction.csv', sep = ',', header=None).to_numpy()
if data_type != 'AI' and data_type != 'FPF': chi = pd.read_csv(file_prefix + data_tag + '-dissipation-rates.csv', sep = ',', header=None).to_numpy()
if data_type == 'NA-SLF':
    enthalpy_defect = pd.read_csv(file_prefix + data_tag + '-enthalpy-defect.csv', sep = ',', header=None).to_numpy()

for species_to_remove in species_to_remove_list:

    (species_index, ) = np.where(state_space_names==species_to_remove)
    if len(species_index) != 0:
        state_space = np.delete(state_space, np.s_[species_index], axis=1)
        state_space_sources = np.delete(state_space_sources, np.s_[species_index], axis=1)
        state_space_names = np.delete(state_space_names, np.s_[species_index])
    else:
        pass

target_variables = state_space[:,target_variables_indices]
target_variables_names = list(state_space_names[target_variables_indices])
(_, n_target_variables) = np.shape(target_variables)
print('\nUsing: ' + ', '.join(target_variables_names) + ' as target state variables at the decoder output.\n')

try:
    VD_target_variables = state_space[:,VD_target_variables_indices]
    VD_target_variables_names = list(state_space_names[VD_target_variables_indices])
    print('\nUsing: ' + ', '.join(VD_target_variables_names) + ' as target state variables for VarianceData computation.\n')
except:
    pass

#################################################################################################################################
## Remove pure stream components
#################################################################################################################################

if pure_streams:
    species_to_remove_list = ['N2', 'AR', 'HE']
    pure_streams_prefix = 'tps' # Trainable Pure Streams
else:
    species_to_remove_list = ['N2', 'AR', 'HE', 'H2', 'NH3', 'O2']
    pure_streams_prefix = 'ntps' # Non-Trainable Pure Streams

state_space = pd.read_csv(file_prefix + data_tag + '-state-space.csv', sep = ',', header=None).to_numpy()
state_space_sources = pd.read_csv(file_prefix + data_tag + '-state-space-sources.csv', sep = ',', header=None).to_numpy()
state_space_names = pd.read_csv(file_prefix + data_tag + '-state-space-names.csv', sep = ',', header=None).to_numpy().ravel()
tex_names = pd.read_csv('../data/ammonia-Stagni-tex-names.csv', sep = ',', header=None).to_numpy().ravel()

for species_to_remove in species_to_remove_list:

    (species_index, ) = np.where(state_space_names==species_to_remove)
    if len(species_index) != 0:
        state_space = np.delete(state_space, np.s_[species_index], axis=1)
        state_space_sources = np.delete(state_space_sources, np.s_[species_index], axis=1)
        state_space_names = np.delete(state_space_names, np.s_[species_index])
        tex_names = np.delete(tex_names, np.s_[species_index])
    else:
        pass

#################################################################################################################################
## Sample data to speed-up tests
#################################################################################################################################

if data_type == 'SLF' or data_type == 'STLF' or data_type == 'NA-SLF':

    idx_reaction_zone, _ = np.where(mf<0.3)
    state_space = state_space[idx_reaction_zone,:]
    state_space_sources = state_space_sources[idx_reaction_zone,:]
    target_variables = target_variables[idx_reaction_zone,:]
    mf = mf[idx_reaction_zone,:]
    chi = chi[idx_reaction_zone,:]
    if data_type == 'NA-SLF': enthalpy_defect = enthalpy_defect[idx_reaction_zone,:]

    try:
        VD_target_variables = VD_target_variables[idx_reaction_zone,:]
    except:
        pass

elif data_type == 'FPF':
    
    idx_trim, _ =  np.where((grid>0.005)&(grid <0.015))
    state_space = state_space[idx_trim,:]
    state_space_sources = state_space_sources[idx_trim,:]
    target_variables = target_variables[idx_trim,:]
    grid = grid[idx_trim,:]
    mf = mf[idx_trim,:]

    try:
        VD_target_variables = VD_target_variables[idx_trim,:]
    except:
        pass

# Take only two enthalpy planes:
if data_type == 'NA-SLF':
    unique_enthalpy_planes = np.unique(enthalpy_defect)
    idx_enthalpy, _ =  np.where((enthalpy_defect==unique_enthalpy_planes[0]) | (enthalpy_defect==unique_enthalpy_planes[-1]))
    state_space = state_space[idx_enthalpy,:]
    state_space_sources = state_space_sources[idx_enthalpy,:]
    target_variables = target_variables[idx_enthalpy,:]
    mf = mf[idx_enthalpy,:]
    chi = chi[idx_enthalpy,:]
    enthalpy_defect = enthalpy_defect[idx_enthalpy,:]

    try:
        VD_target_variables = VD_target_variables[idx_enthalpy,:]
    except:
        pass

#################################################################################################################################
## Print data shape
#################################################################################################################################

(n_observations, n_variables) = np.shape(state_space)

print(str(n_observations) + ' observations')
print(str(n_variables) + ' state variables')
    
#################################################################################################################################