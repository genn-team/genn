

//-------------------------------------------------------------------------
/*! \brief Function to (re)set all model variables to their compile-time, homogeneous initial values.
 Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device.
*/
//-------------------------------------------------------------------------

void initialize()
 {
    
    #if defined(__GNUG__) && !defined(__clang__) && defined(__x86_64__) && __GLIBC__ == 2 && (__GLIBC_MINOR__ == 23 || __GLIBC_MINOR__ == 24)
    if(std::getenv("LD_BIND_NOW") == NULL) {
        fprintf(stderr, "Warning: a bug has been found in glibc 2.23 or glibc 2.24 (https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280) "
                        "which results in poor CPU maths performance. We recommend setting the environment variable LD_BIND_NOW=1 to work around this issue.\n");
    }
    #endif
    srand((unsigned int) time(NULL));
     {
        uint32_t seedData[std::mt19937::state_size];
        std::random_device seedSource;
         {
            for(int i = 0; i < std::mt19937::state_size; i++) {
                seedData[i] = seedSource();
            }
        }
        std::seed_seq seeds(std::begin(seedData), std::end(seedData));
        rng.seed(seeds);
    }
    
    // neuron variables
        spkQuePtrExcitatory = 0;
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_spkQuePtrExcitatory, &spkQuePtrExcitatory, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
     {
        for (int i = 0; i < 2; i++) {
            glbSpkCntExcitatory[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 6400; i++) {
            glbSpkExcitatory[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 3200; i++) {
            t_spikeExcitatory[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 3200; i++) {
            const scalar scale = (-5.00000000000000000e+01f) - (-6.00000000000000000e+01f);
            vExcitatory[i] = (-6.00000000000000000e+01f) + (standardUniformDistribution(rng) * scale);
        }
    }
    
     {
        for (int i = 0; i < 3200; i++) {
            _regimeIDExcitatory[i] = (0.00000000000000000e+00f);
        }
    }
    
        spkQuePtrInhibitory = 0;
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_spkQuePtrInhibitory, &spkQuePtrInhibitory, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
     {
        for (int i = 0; i < 2; i++) {
            glbSpkCntInhibitory[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 1600; i++) {
            glbSpkInhibitory[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 800; i++) {
            t_spikeInhibitory[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 800; i++) {
            const scalar scale = (-5.00000000000000000e+01f) - (-6.00000000000000000e+01f);
            vInhibitory[i] = (-6.00000000000000000e+01f) + (standardUniformDistribution(rng) * scale);
        }
    }
    
     {
        for (int i = 0; i < 800; i++) {
            _regimeIDInhibitory[i] = (0.00000000000000000e+00f);
        }
    }
    
    glbSpkCntSpike_Source[0] = 0;
     {
        for (int i = 0; i < 20; i++) {
            glbSpkSpike_Source[i] = 0;
        }
    }
    
    
    // synapse variables
     {
        for (int i = 0; i < 3200; i++) {
            inSynExcitatory_to_Excitatory_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 800; i++) {
            inSynExcitatory_to_Inhibitory_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 3200; i++) {
            inSynInhibitory_to_Excitatory_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 800; i++) {
            inSynInhibitory_to_Inhibitory_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 3200; i++) {
            inSynSpike_Source_to_Excitatory_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 800; i++) {
            inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
    
    
}

void initializeAllSparseArrays()
 {
    
    initializeSparseArray(CExcitatory_to_Excitatory_Synapse_0_weight_update, d_indExcitatory_to_Excitatory_Synapse_0_weight_update, d_indInGExcitatory_to_Excitatory_Synapse_0_weight_update, 3200);
    initializeSparseArray(CExcitatory_to_Inhibitory_Synapse_0_weight_update, d_indExcitatory_to_Inhibitory_Synapse_0_weight_update, d_indInGExcitatory_to_Inhibitory_Synapse_0_weight_update, 3200);
    initializeSparseArray(CInhibitory_to_Excitatory_Synapse_0_weight_update, d_indInhibitory_to_Excitatory_Synapse_0_weight_update, d_indInGInhibitory_to_Excitatory_Synapse_0_weight_update, 800);
    initializeSparseArray(CInhibitory_to_Inhibitory_Synapse_0_weight_update, d_indInhibitory_to_Inhibitory_Synapse_0_weight_update, d_indInGInhibitory_to_Inhibitory_Synapse_0_weight_update, 800);
    initializeSparseArray(CSpike_Source_to_Excitatory_Synapse_0_weight_update, d_indSpike_Source_to_Excitatory_Synapse_0_weight_update, d_indInGSpike_Source_to_Excitatory_Synapse_0_weight_update, 20);
    initializeSparseArray(CSpike_Source_to_Inhibitory_Synapse_0_weight_update, d_indSpike_Source_to_Inhibitory_Synapse_0_weight_update, d_indInGSpike_Source_to_Inhibitory_Synapse_0_weight_update, 20);
}


void initmodel()
 {
    
    
    
    copyStateToDevice(true);
    
    initializeAllSparseArrays();
}

