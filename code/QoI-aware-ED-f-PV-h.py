#################################################################################################################################

n_output_variables = len(target_variables_indices) + 1

encoder_inputs = np.hstack((Yi_CS, mf/np.max(mf), enthalpy_defect/np.max(np.abs(enthalpy_defect))))

sample_random = preprocess.DataSampler(np.zeros((n_observations,)).astype(int), random_seed=master_random_seed, verbose=False)
(idx_train, idx_validation) = sample_random.random(100 - validation_perc)

batch_size = len(idx_train)

for random_seed in random_seeds_list:

    #################################################################################################################################
    ## Track this run with Weights & Biases
    #################################################################################################################################

    if log_wandb:
    
        wandb.init(project=wandb_project_name,
                   name=case_run_name + '-' + str(random_seed),
                   config={'parameterization': parameterization,
                           'case_name': case_name,
                           'data_tag': data_tag,
                           'data_type': data_type,
                           'target_variables_indices': target_variables_indices,
                           'pure_streams': pure_streams,
                           'n_decoder_arch': (n_decoder_1, n_decoder_2, n_decoder_3),
                           'n_epochs': n_epochs,
                           'init_lr': init_lr,
                           'alpha_lr': alpha_lr,
                           'n_decay_steps': n_decay_steps,
                           'validation_perc': validation_perc,
                           'scaling': scaling,
                           'initializer': initializer,
                           'rmsprop_rho': rmsprop_rho,
                           'rmsprop_momentum': rmsprop_momentum,
                           'rmsprop_centered': rmsprop_centered,
                           'random_seeds_tuple': (min_random_seed, max_random_seed)})

    model_weights_filename = 'f-PV-h-' + case_run_name + '-best-model-weights-rs-' + str(random_seed) + '.h5'
     
    if not os.path.exists('f-PV-h-' + case_run_name + '-MSE-training-losses-rs-' + str(random_seed) + '.csv'):

        print('Computing random seed: ' + str(random_seed))
        
        #################################################################################################################################
        ## Prepare an encoder-decoder
        #################################################################################################################################
    
        tf.random.set_seed(random_seed)
        tf.keras.utils.set_random_seed(random_seed)

        if initializer == 'RandomNormal':
            kernel_initializer = tf.keras.initializers.RandomNormal(seed=random_seed)
        elif initializer == 'RandomUniform':
            kernel_initializer = tf.keras.initializers.RandomUniform(seed=random_seed)
        elif initializer == 'GlorotNormal':
            kernel_initializer = tf.keras.initializers.GlorotNormal(seed=random_seed)
        elif initializer == 'GlorotUniform':
            kernel_initializer = tf.keras.initializers.GlorotUniform(seed=random_seed)
        elif initializer == 'HeNormal':
            kernel_initializer = tf.keras.initializers.HeNormal(seed=random_seed)
        elif initializer == 'HeUniform':
            kernel_initializer = tf.keras.initializers.HeUniform(seed=random_seed)
        elif initializer == 'LecunNormal':
            kernel_initializer = tf.keras.initializers.LecunNormal(seed=random_seed)
        elif initializer == 'LecunUniform':
            kernel_initializer = tf.keras.initializers.LecunUniform(seed=random_seed)
            
        # RMSprop optimizer:
        model_optimizer = tf.optimizers.legacy.RMSprop(learning_rate=learning_rate,
                                                       rho=rmsprop_rho,
                                                       momentum=rmsprop_momentum,
                                                       centered=rmsprop_centered)

        input_layer = Input(shape=(n_variables+1,)) # it is equal to n_variables + 1 because there's n_variables - 1 mass fractions, a mixture fraction, and an enthalpy defect
        input_Yi = Lambda(lambda x: x[:,0:n_variables-1])(input_layer) # mass fractions
        input_f = Lambda(lambda x: x[:,n_variables-1:n_variables])(input_layer) # mixture fraction
        input_h = Lambda(lambda x: x[:,n_variables:n_variables+1])(input_layer) # enthalpy defect
        
        # Assemble the encoding layer:
        encoder_PV = Dense(1, activation='linear', use_bias=False, kernel_initializer=kernel_initializer)(input_Yi)
        encoder_f = Dense(1, activation='linear', use_bias=False, trainable=False, kernel_initializer=tf.keras.initializers.Ones())(input_f)
        encoder_h = Dense(1, activation='linear', use_bias=False, trainable=False, kernel_initializer=tf.keras.initializers.Ones())(input_h)
        encoder_output = concatenate([encoder_PV, encoder_f, encoder_h])
        
        # Create decoding layers:
        decoder_1 = Dense(n_output_variables + n_decoder_1, activation='tanh', kernel_initializer=kernel_initializer)(encoder_output)
        decoder_2 = Dense(n_output_variables + n_decoder_2, activation='tanh', kernel_initializer=kernel_initializer)(decoder_1)
        decoder_3 = Dense(n_output_variables+ n_decoder_3, activation='tanh', kernel_initializer=kernel_initializer)(decoder_2)
        output_layer = Dense(n_output_variables, activation='tanh', kernel_initializer=kernel_initializer)(decoder_3)
        
        model = Model(input_layer, output_layer)
        model.compile(model_optimizer, loss=tf.keras.losses.MeanSquaredError())

        # Rescale the weights for the PV:
        weights_and_biases = model.get_weights()
        current_PV = np.dot(Yi_CS, weights_and_biases[0])
        weight_scaling_factor = np.max(current_PV) - np.min(current_PV)
        weights_and_biases[0] = weights_and_biases[0] / weight_scaling_factor
        model.set_weights(weights_and_biases)

        # Debug:
        weights_and_biases = model.get_weights()
        current_PV = np.dot(Yi_CS, weights_and_biases[0])
        print(np.max(current_PV) - np.min(current_PV))
        
        # Initialize the output variables:
        PV_source = np.dot(Yi_sources_CS, model.get_weights()[0])
        decoder_outputs = np.hstack((target_variables, PV_source))
        decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='-1to1')

        # Determine the initial validation data:
        validation_data = (encoder_inputs[idx_validation,:], decoder_outputs_normalized[idx_validation,:])
        
        #################################################################################################################################
        ## Train an encoder-decoder
        #################################################################################################################################
        
        tic = time.perf_counter()
    
        weights_and_biases_init = model.get_weights()
        training_losses_across_epochs = []
        validation_losses_across_epochs = []
        epochs_list = [e for e in range(0, n_epochs)]
        previous_best_training_loss = model.evaluate(encoder_inputs[idx_train,:], decoder_outputs_normalized[idx_train,:], verbose=0)
        validation_losses_across_epochs.append(model.evaluate(encoder_inputs[idx_validation,:], decoder_outputs_normalized[idx_validation,:], verbose=0))

        # Log metrics to WandB:
        if log_wandb: wandb.log({"validation-MSE-loss": validation_losses_across_epochs[0]})

        for i_epoch in tqdm(epochs_list):

            # Capture weights prior to optimization:
            weights_prior_to_optimization = model.get_weights()
            
            history = model.fit(encoder_inputs[idx_train,:],
                                decoder_outputs_normalized[idx_train,:],
                                epochs=1,
                                batch_size=batch_size,
                                validation_data=validation_data,
                                verbose=0)
            
            # Rescale the weights for the PV:
            weights_and_biases = model.get_weights()
            current_PV = np.dot(Yi_CS, weights_and_biases[0])
            weight_scaling_factor = np.max(current_PV) - np.min(current_PV)
            weights_and_biases[0] = weights_and_biases[0] / weight_scaling_factor
            model.set_weights(weights_and_biases)

            # Update the projection-dependent output variables for the next epoch:
            PV_source = np.dot(Yi_sources_CS, model.get_weights()[0])
            decoder_outputs = np.hstack((target_variables, PV_source))
            decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='-1to1')
        
            # Update the validation data for the next epoch:
            validation_data = (encoder_inputs[idx_validation,:], decoder_outputs_normalized[idx_validation,:])

            # Save current epoch:
            training_losses_across_epochs.append(history.history['loss'][0]) # Training loss corresponds to weights before optimization
            validation_losses_across_epochs.append(history.history['val_loss'][0])

            # Log metrics to WandB:
            if log_wandb:
                wandb.log({"train-MSE-loss": history.history['loss'][0],
                           "validation-MSE-loss": history.history['val_loss'][0]})

            # Save the whole trained network corresponding to the so far best training loss:
            if (i_epoch > 0) and (training_losses_across_epochs[-1] < previous_best_training_loss):

                with h5py.File(model_weights_filename, 'w', libver='latest') as f:
                    for idx, arr in enumerate(weights_prior_to_optimization):    
                        dset = f.create_dataset(str(idx), data=arr, compression='gzip', compression_opts=9)
                    f.close()
                    
                previous_best_training_loss = training_losses_across_epochs[-1]
                best_epoch_counter = i_epoch
                
        # Append training and validation loss corresponding to the final weights in the trained network:
        training_losses_across_epochs.append(model.evaluate(encoder_inputs[idx_train,:], decoder_outputs_normalized[idx_train,:], verbose=0))

        # Log metrics to WandB:
        if log_wandb:
            wandb.log({"train-MSE-loss": training_losses_across_epochs[-1]})
            wandb.finish()

        if training_losses_across_epochs[-1] < previous_best_training_loss:

            with h5py.File(model_weights_filename, 'w', libver='latest') as f:
                for idx, arr in enumerate(model.get_weights()):    
                    dset = f.create_dataset(str(idx), data=arr, compression='gzip', compression_opts=9)
                f.close()
            previous_best_training_loss = training_losses_across_epochs[-1]
            print('Best basis is the final one')
            best_epoch_counter += 1
        
        else:
        
            print('Best basis at epoch: ' + str(best_epoch_counter))

        np.savetxt('f-PV-h-' + case_run_name + '-best-epoch-counter-rs-' + str(random_seed) + '.csv', ([best_epoch_counter]), delimiter=',', fmt='%.16e')
        np.savetxt('f-PV-h-' + case_run_name + '-previous-best-training-loss-rs-' + str(random_seed) + '.csv', ([previous_best_training_loss]), delimiter=',', fmt='%.16e')

        toc = time.perf_counter()
        print(f'Time it took: {(toc - tic)/60:0.1f} minutes.\n')
    
        #################################################################################################################################
        ## Save loss
        #################################################################################################################################
    
        np.savetxt('f-PV-h-' + case_run_name + '-MSE-training-losses-rs-' + str(random_seed) + '.csv',(training_losses_across_epochs), delimiter=',', fmt='%.16e')
        np.savetxt('f-PV-h-' + case_run_name + '-MSE-validation-losses-rs-' + str(random_seed) + '.csv',(validation_losses_across_epochs), delimiter=',', fmt='%.16e')
        
        #################################################################################################################################
        ## Clear this Keras session
        #################################################################################################################################
        
        K.clear_session()
    
    else:

        print('Random seed ' + str(random_seed) + ' already done, moving on!')
        
################################################################################################################################