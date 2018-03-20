

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
    glbSpkCntHH[0] = 0;
     {
        for (int i = 0; i < 12; i++) {
            glbSpkHH[i] = 0;
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            VHH[i] = (-6.00000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            mHH[i] = (5.29323999999999975e-02f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            hHH[i] = (3.17676699999999979e-01f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            nHH[i] = (5.96120699999999948e-01f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            gNaHH[i] = (1.20000000000000000e+02f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            ENaHH[i] = (5.50000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            gKHH[i] = (3.60000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            EKHH[i] = (-7.20000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            glHH[i] = (2.99999999999999989e-01f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            ElHH[i] = (-5.00000000000000000e+01f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            CHH[i] = (1.00000000000000000e+00f);
        }
    }
    
     {
        for (int i = 0; i < 12; i++) {
            errHH[i] = (0.00000000000000000e+00f);
        }
    }
    
    
    // synapse variables
    
    
    copyStateToDevice(true);
    
}

void initializeAllSparseArrays()
 {
    
}


void initHHVClamp()
 {
    
    
    
}

