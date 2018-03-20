

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
        spkQuePtrInput = 0;
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_spkQuePtrInput, &spkQuePtrInput, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
     {
        for (int i = 0; i < 7; i++) {
            glbSpkCntInput[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 3500; i++) {
            glbSpkInput[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 500; i++) {
            VInput[i] = (-6.50000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 500; i++) {
            UInput[i] = (-2.00000000000000000e+01f);
        }
    }
    
    glbSpkCntInter[0] = 0;
     {
        for (int i = 0; i < 500; i++) {
            glbSpkInter[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 500; i++) {
            VInter[i] = (-6.50000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 500; i++) {
            UInter[i] = (-2.00000000000000000e+01f);
        }
    }
    
    glbSpkCntOutput[0] = 0;
     {
        for (int i = 0; i < 500; i++) {
            glbSpkOutput[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 500; i++) {
            VOutput[i] = (-6.50000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 500; i++) {
            UOutput[i] = (-2.00000000000000000e+01f);
        }
    }
    
    
    // synapse variables
     {
        for (int i = 0; i < 500; i++) {
            inSynInputInter[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 500; i++) {
            inSynInputOutput[i] = 0.000000f;
        }
    }
    
     {
        for (int i = 0; i < 500; i++) {
            inSynInterOutput[i] = 0.000000f;
        }
    }
    
    
    
    copyStateToDevice(true);
    
}

void initializeAllSparseArrays()
 {
    
}


void initSynDelay()
 {
    
    
    
}

