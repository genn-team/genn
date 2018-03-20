

extern "C" __global__ void initializeDevice()
 {
    const unsigned int id = 96 * blockIdx.x + threadIdx.x;
    // Initialise global GPU RNG
    if(id == 0) {
        curand_init(0, 0, 0, &dd_rng[0]);
    }
    
    // neuron group PExc
    if (id < 8064) {
        const unsigned int lid = id - 0;
        if(lid == 0) {
            dd_glbSpkCntPExc[0] = 0;
        }
        // only do this for existing neurons
        if (lid < 8000) {
            curand_init(0, id, 0, &dd_rngPExc[lid]);
            curandStatePhilox4_32_10_t initRNG = dd_rng[0];
            skipahead_sequence((unsigned long long)id, &initRNG);
            dd_glbSpkPExc[lid] = 0;
             {
                dd_VPExc[lid] = (-6.50000000000000000e+01f);
            }
             {
                dd_UPExc[lid] = (-1.30000000000000000e+01f);
            }
             {
                dd_aPExc[lid] = (2.00000000000000004e-02f);
            }
             {
                dd_bPExc[lid] = (2.00000000000000011e-01f);
            }
             {
                const scalar random = curand_uniform(&initRNG);
                dd_cPExc[lid] = (-6.50000000000000000e+01f) + (random * random * (1.50000000000000000e+01f));
            }
             {
                const scalar random = curand_uniform(&initRNG);
                dd_dPExc[lid] = (8.00000000000000000e+00f) + (random * random * (-6.00000000000000000e+00f));
            }
            dd_inSynExc_Exc[lid] = 0.000000f;
            dd_inSynInh_Exc[lid] = 0.000000f;
        }
    }
    // neuron group PInh
    if ((id >= 8064) && (id < 10080)) {
        const unsigned int lid = id - 8064;
        if(lid == 0) {
            dd_glbSpkCntPInh[0] = 0;
        }
        // only do this for existing neurons
        if (lid < 2000) {
            curand_init(0, id, 0, &dd_rngPInh[lid]);
            curandStatePhilox4_32_10_t initRNG = dd_rng[0];
            skipahead_sequence((unsigned long long)id, &initRNG);
            dd_glbSpkPInh[lid] = 0;
             {
                const scalar scale = (1.00000000000000006e-01f) - (2.00000000000000004e-02f);
                dd_aPInh[lid] = (2.00000000000000004e-02f) + (curand_uniform(&initRNG) * scale);
            }
             {
                dd_cPInh[lid] = (-6.50000000000000000e+01f);
            }
             {
                dd_dPInh[lid] = (2.00000000000000000e+00f);
            }
            dd_inSynExc_Inh[lid] = 0.000000f;
            dd_inSynInh_Inh[lid] = 0.000000f;
        }
    }
}
extern "C" __global__ void initializeSparseDevice(unsigned int endThreadExc_Exc, unsigned int numSynapsesExc_Exc, unsigned int endThreadExc_Inh, unsigned int numSynapsesExc_Inh, unsigned int endThreadInh_Exc, unsigned int numSynapsesInh_Exc, unsigned int endThreadInh_Inh, unsigned int numSynapsesInh_Inh)
 {
    const unsigned int id = 128 * blockIdx.x + threadIdx.x;
    // synapse group Exc_Exc
    if (id < endThreadExc_Exc) {
        const unsigned int lid = id;
        // only do this for existing synapses
        if (lid < numSynapsesExc_Exc) {
        }
    }
    // synapse group Exc_Inh
    if ((id >= endThreadExc_Exc) && (id < endThreadExc_Inh)) {
        const unsigned int lid = id - endThreadExc_Exc;
        // only do this for existing synapses
        if (lid < numSynapsesExc_Inh) {
        }
    }
    // synapse group Inh_Exc
    if ((id >= endThreadExc_Inh) && (id < endThreadInh_Exc)) {
        const unsigned int lid = id - endThreadExc_Inh;
        // only do this for existing synapses
        if (lid < numSynapsesInh_Exc) {
        }
    }
    // synapse group Inh_Inh
    if ((id >= endThreadInh_Exc) && (id < endThreadInh_Inh)) {
        const unsigned int lid = id - endThreadInh_Exc;
        // only do this for existing synapses
        if (lid < numSynapsesInh_Inh) {
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
     {
        for (int i = 0; i < 2000; i++) {
            VPInh[i] = (-6.50000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 2000; i++) {
            const scalar scale = (2.50000000000000000e-01f) - (2.00000000000000011e-01f);
            bPInh[i] = (2.00000000000000011e-01f) + (standardUniformDistribution(rng) * scale);
        }
    }
    
    
    // synapse variables
    
    
    // perform on-device init
    dim3 iThreads(96, 1);
    dim3 iGrid(105, 1);
    initializeDevice <<<iGrid, iThreads>>>();
}

void initializeAllSparseArrays()
 {
    
    initializeSparseArray(CExc_Exc, d_indExc_Exc, d_indInGExc_Exc, 8000);
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gExc_Exc, gExc_Exc, sizeof(scalar) * CExc_Exc.connN , cudaMemcpyHostToDevice));
    initializeSparseArray(CExc_Inh, d_indExc_Inh, d_indInGExc_Inh, 8000);
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gExc_Inh, gExc_Inh, sizeof(scalar) * CExc_Inh.connN , cudaMemcpyHostToDevice));
    initializeSparseArray(CInh_Exc, d_indInh_Exc, d_indInGInh_Exc, 2000);
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gInh_Exc, gInh_Exc, sizeof(scalar) * CInh_Exc.connN , cudaMemcpyHostToDevice));
    initializeSparseArray(CInh_Inh, d_indInh_Inh, d_indInGInh_Inh, 2000);
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gInh_Inh, gInh_Inh, sizeof(scalar) * CInh_Inh.connN , cudaMemcpyHostToDevice));
}


void initIzh_sparse()
 {
    
    
    
    copyStateToDevice(true);
    
    initializeAllSparseArrays();
     {
        // Calculate block sizes based on number of connections in sparse projection
        const unsigned int endThreadExc_Exc = (unsigned int)(ceil((double)CExc_Exc.connN / (double)128) * (double)128);
        const unsigned int endThreadExc_Inh = (unsigned int)(ceil((double)CExc_Inh.connN / (double)128) * (double)128) + endThreadExc_Exc;
        const unsigned int endThreadInh_Exc = (unsigned int)(ceil((double)CInh_Exc.connN / (double)128) * (double)128) + endThreadExc_Inh;
        const unsigned int endThreadInh_Inh = (unsigned int)(ceil((double)CInh_Inh.connN / (double)128) * (double)128) + endThreadInh_Exc;
        // perform on-device sparse init
        dim3 iThreads(128, 1);
        dim3 iGrid(endThreadInh_Inh / 128, 1);
        initializeSparseDevice <<<iGrid, iThreads>>>(endThreadExc_Exc, CExc_Exc.connN, endThreadExc_Inh, CExc_Inh.connN, endThreadInh_Exc, CInh_Exc.connN, endThreadInh_Inh, CInh_Inh.connN);
    }
}

