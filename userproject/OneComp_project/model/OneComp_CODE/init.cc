

extern "C" __global__ void initializeDevice()
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // neuron group Izh1
    if (id < 32) {
        const unsigned int lid = id - 0;
        if(lid == 0) {
            dd_glbSpkCntIzh1[0] = 0;
        }
        // only do this for existing neurons
        if (lid < 1) {
            dd_glbSpkIzh1[lid] = 0;
             {
                dd_VIzh1[lid] = (-6.50000000000000000e+01f);
            }
             {
                dd_UIzh1[lid] = (-2.00000000000000000e+01f);
            }
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
    
    // neuron variables
    
    // synapse variables
    
    
    copyStateToDevice(true);
    
    // perform on-device init
    dim3 iThreads(32, 1);
    dim3 iGrid(1, 1);
    initializeDevice <<<iGrid, iThreads>>>();
}

void initializeAllSparseArrays()
 {
    
}


void initOneComp()
 {
    
    
    
}

