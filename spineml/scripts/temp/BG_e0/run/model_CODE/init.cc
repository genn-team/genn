

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
    
    // neuron variables
    glbSpkCntCortex[0] = 0;
     {
        for (int i = 0; i < 6; i++) {
            glbSpkCortex[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            aCortex[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inCortex[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            outCortex[i] = (0.00000000000000000e+00f);
        }
    }
    
    glbSpkCntD1[0] = 0;
     {
        for (int i = 0; i < 6; i++) {
            glbSpkD1[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            aD1[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            outD1[i] = (0.00000000000000000e+00f);
        }
    }
    
    glbSpkCntD2[0] = 0;
     {
        for (int i = 0; i < 6; i++) {
            glbSpkD2[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            aD2[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            outD2[i] = (0.00000000000000000e+00f);
        }
    }
    
        spkQuePtrGPe = 0;
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_spkQuePtrGPe, &spkQuePtrGPe, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    glbSpkCntGPe[0] = 0;
     {
        for (int i = 0; i < 6; i++) {
            glbSpkGPe[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            aGPe[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            outGPe[i] = (0.00000000000000000e+00f);
        }
    }
    
    glbSpkCntGPi[0] = 0;
     {
        for (int i = 0; i < 6; i++) {
            glbSpkGPi[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            aGPi[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            outGPi[i] = (0.00000000000000000e+00f);
        }
    }
    
    glbSpkCntSTN[0] = 0;
     {
        for (int i = 0; i < 6; i++) {
            glbSpkSTN[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            aSTN[i] = (0.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            outSTN[i] = (0.00000000000000000e+00f);
        }
    }
    
    
    // synapse variables
     {
        for (int i = 0; i < 6; i++) {
            inSynCortex_to_D1_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inSynCortex_to_D2_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inSynCortex_to_STN_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inSynD1_to_GPi_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inSynD2_to_GPe_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inSynGPe_to_GPi_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inSynGPe_to_STN_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inSynSTN_to_GPe_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 6; i++) {
            inSynSTN_to_GPi_Synapse_0_weight_update[i] = 0.000000f;
        }
    }
    
    
    
}

void initializeAllSparseArrays()
 {
    
    initializeSparseArray(CCortex_to_D1_Synapse_0_weight_update, d_indCortex_to_D1_Synapse_0_weight_update, d_indInGCortex_to_D1_Synapse_0_weight_update, 6);
    initializeSparseArrayPreInd(CCortex_to_D1_Synapse_0_weight_update, d_preIndCortex_to_D1_Synapse_0_weight_update);
    initializeSparseArray(CCortex_to_D2_Synapse_0_weight_update, d_indCortex_to_D2_Synapse_0_weight_update, d_indInGCortex_to_D2_Synapse_0_weight_update, 6);
    initializeSparseArrayPreInd(CCortex_to_D2_Synapse_0_weight_update, d_preIndCortex_to_D2_Synapse_0_weight_update);
    initializeSparseArray(CCortex_to_STN_Synapse_0_weight_update, d_indCortex_to_STN_Synapse_0_weight_update, d_indInGCortex_to_STN_Synapse_0_weight_update, 6);
    initializeSparseArrayPreInd(CCortex_to_STN_Synapse_0_weight_update, d_preIndCortex_to_STN_Synapse_0_weight_update);
    initializeSparseArray(CD1_to_GPi_Synapse_0_weight_update, d_indD1_to_GPi_Synapse_0_weight_update, d_indInGD1_to_GPi_Synapse_0_weight_update, 6);
    initializeSparseArrayPreInd(CD1_to_GPi_Synapse_0_weight_update, d_preIndD1_to_GPi_Synapse_0_weight_update);
    initializeSparseArray(CD2_to_GPe_Synapse_0_weight_update, d_indD2_to_GPe_Synapse_0_weight_update, d_indInGD2_to_GPe_Synapse_0_weight_update, 6);
    initializeSparseArrayPreInd(CD2_to_GPe_Synapse_0_weight_update, d_preIndD2_to_GPe_Synapse_0_weight_update);
    initializeSparseArray(CGPe_to_GPi_Synapse_0_weight_update, d_indGPe_to_GPi_Synapse_0_weight_update, d_indInGGPe_to_GPi_Synapse_0_weight_update, 6);
    initializeSparseArrayPreInd(CGPe_to_GPi_Synapse_0_weight_update, d_preIndGPe_to_GPi_Synapse_0_weight_update);
    initializeSparseArray(CGPe_to_STN_Synapse_0_weight_update, d_indGPe_to_STN_Synapse_0_weight_update, d_indInGGPe_to_STN_Synapse_0_weight_update, 6);
    initializeSparseArrayPreInd(CGPe_to_STN_Synapse_0_weight_update, d_preIndGPe_to_STN_Synapse_0_weight_update);
}


void initmodel()
 {
    
    createPreIndices(6, 6, &CCortex_to_D1_Synapse_0_weight_update);
    createPreIndices(6, 6, &CCortex_to_D2_Synapse_0_weight_update);
    createPreIndices(6, 6, &CCortex_to_STN_Synapse_0_weight_update);
    createPreIndices(6, 6, &CD1_to_GPi_Synapse_0_weight_update);
    createPreIndices(6, 6, &CD2_to_GPe_Synapse_0_weight_update);
    createPreIndices(6, 6, &CGPe_to_GPi_Synapse_0_weight_update);
    createPreIndices(6, 6, &CGPe_to_STN_Synapse_0_weight_update);
    
    
    copyStateToDevice(true);
    
    initializeAllSparseArrays();
}

