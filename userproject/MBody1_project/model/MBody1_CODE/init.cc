

extern "C" __global__ void initializeDevice()
 {
    const unsigned int id = 128 * blockIdx.x + threadIdx.x;
    // neuron group DN
    if (id < 128) {
        const unsigned int lid = id - 0;
        if(lid == 0) {
            dd_glbSpkCntDN[0] = 0;
            dd_glbSpkCntEvntDN[0] = 0;
        }
        // only do this for existing neurons
        if (lid < 100) {
            dd_glbSpkDN[lid] = 0;
            dd_glbSpkCntDN[lid] = 0;
            dd_sTDN[lid] = -SCALAR_MAX;
             {
                scalar initVal;
                initVal = (-6.00000000000000000e+01f);
                for (int i = 0; i < 1; i++) {
                    dd_VDN[(i * 100) + lid] = initVal;
                }
                
            }
             {
                dd_mDN[lid] = (5.29323999999999975e-02f);
            }
             {
                dd_hDN[lid] = (3.17676699999999979e-01f);
            }
             {
                dd_nDN[lid] = (5.96120699999999948e-01f);
            }
            dd_inSynKCDN[lid] = 0.000000f;
            dd_inSynDNDN[lid] = 0.000000f;
        }
    }
    // neuron group KC
    if ((id >= 128) && (id < 1152)) {
        const unsigned int lid = id - 128;
        if(lid == 0) {
            dd_glbSpkCntKC[0] = 0;
        }
        // only do this for existing neurons
        if (lid < 1000) {
            dd_glbSpkKC[lid] = 0;
            dd_sTKC[lid] = -SCALAR_MAX;
             {
                dd_VKC[lid] = (-6.00000000000000000e+01f);
            }
             {
                dd_mKC[lid] = (5.29323999999999975e-02f);
            }
             {
                dd_hKC[lid] = (3.17676699999999979e-01f);
            }
             {
                dd_nKC[lid] = (5.96120699999999948e-01f);
            }
            dd_inSynPNKC[lid] = 0.000000f;
            dd_inSynLHIKC[lid] = 0.000000f;
        }
    }
    // neuron group LHI
    if ((id >= 1152) && (id < 1280)) {
        const unsigned int lid = id - 1152;
        if(lid == 0) {
            dd_glbSpkCntLHI[0] = 0;
            dd_glbSpkCntEvntLHI[0] = 0;
        }
        // only do this for existing neurons
        if (lid < 20) {
            dd_glbSpkLHI[lid] = 0;
            dd_glbSpkCntLHI[lid] = 0;
             {
                scalar initVal;
                initVal = (-6.00000000000000000e+01f);
                for (int i = 0; i < 1; i++) {
                    dd_VLHI[(i * 20) + lid] = initVal;
                }
                
            }
             {
                dd_mLHI[lid] = (5.29323999999999975e-02f);
            }
             {
                dd_hLHI[lid] = (3.17676699999999979e-01f);
            }
             {
                dd_nLHI[lid] = (5.96120699999999948e-01f);
            }
            dd_inSynPNLHI[lid] = 0.000000f;
        }
    }
    // neuron group PN
    if ((id >= 1280) && (id < 1408)) {
        const unsigned int lid = id - 1280;
        if(lid == 0) {
            dd_glbSpkCntPN[0] = 0;
        }
        // only do this for existing neurons
        if (lid < 100) {
            dd_glbSpkPN[lid] = 0;
             {
                dd_VPN[lid] = (-6.00000000000000000e+01f);
            }
             {
                dd_seedPN[lid] = (0.00000000000000000e+00f);
            }
             {
                dd_spikeTimePN[lid] = (-1.00000000000000000e+01f);
            }
        }
    }
    // synapse group KCDN
    if ((id >= 1408) && (id < 101504)) {
        const unsigned int lid = id - 1408;
        // only do this for existing synapses
        if (lid < 100000) {
        }
    }
    // synapse group PNKC
    if ((id >= 101504) && (id < 201600)) {
        const unsigned int lid = id - 101504;
        // only do this for existing synapses
        if (lid < 100000) {
        }
    }
    // synapse group PNLHI
    if ((id >= 201600) && (id < 203648)) {
        const unsigned int lid = id - 201600;
        // only do this for existing synapses
        if (lid < 2000) {
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
    
    // neuron variables
     {
        for (int i = 0; i < 100; i++) {
            seedPN[i] = rand();
        }
    }
    
    
    // synapse variables
    
    
    copyStateToDevice(true);
    
    // perform on-device init
    dim3 iThreads(128, 1);
    dim3 iGrid(1591, 1);
    initializeDevice <<<iGrid, iThreads>>>();
}

void initializeAllSparseArrays()
 {
    
}


void initMBody1()
 {
    
    
    
}

