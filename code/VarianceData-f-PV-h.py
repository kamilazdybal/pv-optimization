#################################################################################################################################

depvars_names = VD_target_variables_names + ['PV-source']

tic = time.perf_counter()

print('- '*30)

for random_seed in random_seeds_list:
    
    print('Random seed: ' + str(random_seed) + '...\n')

    tic_variance_data = time.perf_counter()

    model_weights_filename = 'f-PV-h-' + case_run_name + '-best-model-weights-rs-' + str(random_seed) + '.h5'
    variance_data_file = 'f-PV-h-' + case_run_name + '-VD-for-' + '-'.join(VD_target_variables_names) + '-PV-source-rs-' + str(random_seed) + '.pkl'
    
    if not os.path.exists(variance_data_file):
    
        if os.path.exists('f-PV-h-' + case_run_name + '-MSE-training-losses-rs-' + str(random_seed) + '.csv'):
            
            hf = h5py.File(model_weights_filename, 'r')
            best_basis = np.array(hf.get('0'))
            hf.close()
            
            PV = np.dot(Yi_CS, best_basis)
            PV_source = np.dot(Yi_sources_CS, best_basis)

            depvars = np.hstack((VD_target_variables, PV_source))
            indepvars = np.hstack((mf, PV, enthalpy_defect))

            variance_data = analysis.compute_normalized_variance(indepvars,
                                                                 depvars,
                                                                 depvar_names=depvars_names,
                                                                 scale_unit_box=True,
                                                                 bandwidth_values=bandwidth_values)

            pickle.dump(variance_data, open(variance_data_file, "wb" ))

            print('VarianceData computed and saved!')

            toc_variance_data = time.perf_counter()

            print(f'VarianceData computation time: {(toc_variance_data - tic_variance_data)/60:0.1f} minutes.')
            
        else:
            print('Basis for random seed ' + str(random_seed) + ' not computed yet!')
    else:
        print('VarianceData for random seed ' + str(random_seed) + ' is already computed, moving on...')

    print('- '*30)
        
toc = time.perf_counter()

print(f'\n\n\tTotal time it took: {(toc - tic)/60:0.1f} minutes.')
        
#################################################################################################################################