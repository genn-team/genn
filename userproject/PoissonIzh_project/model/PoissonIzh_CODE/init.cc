

extern "C" __global__ void initializeDevice()
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // neuron group PN
    if (id < 128) {
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if (lid < 100) {
            curand_init(1234, id, 0, &dd_rngPN[lid]);
        }
    }
}
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
    srand((unsigned int) 1234);
     {
        std::seed_seq seeds{1234};
        rng.seed(seeds);
    }
    
    // neuron variables
    glbSpkCntIzh1[0] = 0;
     {
        for (int i = 0; i < 10; i++) {
            glbSpkIzh1[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 10; i++) {
            VIzh1[i] = (-6.50000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 10; i++) {
            UIzh1[i] = (-2.00000000000000000e+01f);
        }
    }
    
    glbSpkCntPN[0] = 0;
     {
        for (int i = 0; i < 100; i++) {
            glbSpkPN[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 100; i++) {
            timeStepToSpikePN[i] = (0.00000000000000000e+00f);
        }
    }
    
    
    // synapse variables
     {
        for (int i = 0; i < 10; i++) {
            inSynPNIzh1[i] = 0.000000f;
        }
    }
    
    
    
    copyStateToDevice(true);
    
    // perform on-device init
    dim3 iThreads(32, 1);
    dim3 iGrid(4, 1);
    initializeDevice <<<iGrid, iThreads>>>();
}

void initializeAllSparseArrays()
 {
    
}


void initPoissonIzh()
 {
    
    
    
}

