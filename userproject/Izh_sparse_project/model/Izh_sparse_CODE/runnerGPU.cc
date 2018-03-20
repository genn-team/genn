
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model Izh_sparse containing the host side code for a GPU simulator version.
*/
//-------------------------------------------------------------------------


// software version of atomic add for double precision
__device__ double atomicAddSW(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template<typename RNG>
__device__ float exponentialDistFloat(RNG *rng) {
    float a = 0.0f;
    while (true) {
        float u = curand_uniform(rng);
        const float u0 = u;
        while (true) {
            float uStar = curand_uniform(rng);
            if (u < uStar) {
                return  a + u0;
            }
            u = curand_uniform(rng);
            if (u >= uStar) {
                break;
            }
        }
        a += 1.0f;
    }
}

template<typename RNG>
__device__ double exponentialDistDouble(RNG *rng) {
    double a = 0.0f;
    while (true) {
        double u = curand_uniform_double(rng);
        const double u0 = u;
        while (true) {
            double uStar = curand_uniform_double(rng);
            if (u < uStar) {
                return  a + u0;
            }
            u = curand_uniform_double(rng);
            if (u >= uStar) {
                break;
            }
        }
        a += 1.0;
    }
}

#include "neuronKrnl.cc"
#include "synapseKrnl.cc"
// ------------------------------------------------------------------------
// copying things to device

void pushPExcStateToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VPExc, VPExc, 8000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_UPExc, UPExc, 8000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_aPExc, aPExc, 8000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_bPExc, bPExc, 8000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_cPExc, cPExc, 8000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dPExc, dPExc, 8000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushPExcSpikesToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPExc, glbSpkCntPExc, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPExc, glbSpkPExc, 8000 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushPExcSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushPExcCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPExc, glbSpkCntPExc, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPExc, glbSpkPExc, glbSpkCntPExc[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushPExcCurrentSpikeEventsToDevice()
 {
}

void pushPInhStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VPInh, VPInh, 2000 * sizeof(scalar), cudaMemcpyHostToDevice));
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_UPInh, UPInh, 2000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_aPInh, aPInh, 2000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA_ERRORS(cudaMemcpy(d_bPInh, bPInh, 2000 * sizeof(scalar), cudaMemcpyHostToDevice));
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_cPInh, cPInh, 2000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_dPInh, dPInh, 2000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushPInhSpikesToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPInh, glbSpkCntPInh, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPInh, glbSpkPInh, 2000 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushPInhSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushPInhCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPInh, glbSpkCntPInh, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPInh, glbSpkPInh, glbSpkCntPInh[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushPInhCurrentSpikeEventsToDevice()
 {
}

void pushExc_ExcStateToDevice(bool hostInitialisedOnly)
 {
    const size_t size = CExc_Exc.connN;
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gExc_Exc, gExc_Exc, size * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynExc_Exc, inSynExc_Exc, 8000 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushExc_InhStateToDevice(bool hostInitialisedOnly)
 {
    const size_t size = CExc_Inh.connN;
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gExc_Inh, gExc_Inh, size * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynExc_Inh, inSynExc_Inh, 2000 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushInh_ExcStateToDevice(bool hostInitialisedOnly)
 {
    const size_t size = CInh_Exc.connN;
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gInh_Exc, gInh_Exc, size * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynInh_Exc, inSynInh_Exc, 8000 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushInh_InhStateToDevice(bool hostInitialisedOnly)
 {
    const size_t size = CInh_Inh.connN;
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gInh_Inh, gInh_Inh, size * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynInh_Inh, inSynInh_Inh, 2000 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

// ------------------------------------------------------------------------
// copying things from device

void pullPExcStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VPExc, d_VPExc, 8000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(UPExc, d_UPExc, 8000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(aPExc, d_aPExc, 8000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(bPExc, d_bPExc, 8000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(cPExc, d_cPExc, 8000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(dPExc, d_dPExc, 8000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullPExcSpikeEventsFromDevice()
 {
}

void pullPExcSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPExc, d_glbSpkCntPExc, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPExc, d_glbSpkPExc, glbSpkCntPExc [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullPExcSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullPExcCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPExc, d_glbSpkCntPExc, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPExc, d_glbSpkPExc, glbSpkCntPExc[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullPExcCurrentSpikeEventsFromDevice()
 {
}

void pullPInhStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VPInh, d_VPInh, 2000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(UPInh, d_UPInh, 2000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(aPInh, d_aPInh, 2000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(bPInh, d_bPInh, 2000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(cPInh, d_cPInh, 2000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(dPInh, d_dPInh, 2000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullPInhSpikeEventsFromDevice()
 {
}

void pullPInhSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPInh, d_glbSpkCntPInh, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPInh, d_glbSpkPInh, glbSpkCntPInh [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullPInhSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullPInhCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPInh, d_glbSpkCntPInh, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPInh, d_glbSpkPInh, glbSpkCntPInh[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullPInhCurrentSpikeEventsFromDevice()
 {
}

void pullExc_ExcStateFromDevice()
 {
    size_t size = CExc_Exc.connN;
    CHECK_CUDA_ERRORS(cudaMemcpy(gExc_Exc, d_gExc_Exc, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynExc_Exc, d_inSynExc_Exc, 8000 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullExc_InhStateFromDevice()
 {
    size_t size = CExc_Inh.connN;
    CHECK_CUDA_ERRORS(cudaMemcpy(gExc_Inh, d_gExc_Inh, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynExc_Inh, d_inSynExc_Inh, 2000 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullInh_ExcStateFromDevice()
 {
    size_t size = CInh_Exc.connN;
    CHECK_CUDA_ERRORS(cudaMemcpy(gInh_Exc, d_gInh_Exc, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynInh_Exc, d_inSynInh_Exc, 8000 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullInh_InhStateFromDevice()
 {
    size_t size = CInh_Inh.connN;
    CHECK_CUDA_ERRORS(cudaMemcpy(gInh_Inh, d_gInh_Inh, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynInh_Inh, d_inSynInh_Inh, 2000 * sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice(bool hostInitialisedOnly)
 {
    pushPExcStateToDevice(hostInitialisedOnly);
    pushPExcSpikesToDevice(hostInitialisedOnly);
    pushPInhStateToDevice(hostInitialisedOnly);
    pushPInhSpikesToDevice(hostInitialisedOnly);
    pushExc_ExcStateToDevice(hostInitialisedOnly);
    pushExc_InhStateToDevice(hostInitialisedOnly);
    pushInh_ExcStateToDevice(hostInitialisedOnly);
    pushInh_InhStateToDevice(hostInitialisedOnly);
}

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushPExcSpikesToDevice();
    pushPInhSpikesToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushPExcCurrentSpikesToDevice();
    pushPInhCurrentSpikesToDevice();
}
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushPExcSpikeEventsToDevice();
    pushPInhSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushPExcCurrentSpikeEventsToDevice();
    pushPInhCurrentSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullPExcStateFromDevice();
    pullPExcSpikesFromDevice();
    pullPInhStateFromDevice();
    pullPInhSpikesFromDevice();
    pullExc_ExcStateFromDevice();
    pullExc_InhStateFromDevice();
    pullInh_ExcStateFromDevice();
    pullInh_InhStateFromDevice();
}

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
    pullPExcSpikesFromDevice();
    pullPInhSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
    pullPExcCurrentSpikesFromDevice();
    pullPInhCurrentSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPExc, d_glbSpkCntPExc, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPInh, d_glbSpkCntPInh, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}


// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
    pullPExcSpikeEventsFromDevice();
    pullPInhSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
    pullPExcCurrentSpikeEventsFromDevice();
    pullPInhCurrentSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void copySpikeEventNFromDevice()
 {
    
}


// ------------------------------------------------------------------------
// the time stepping procedure (using GPU)
void stepTimeGPU()
 {
    
    //model.padSumSynapseTrgN[model.synapseGrpN - 1] is 2400
    dim3 sThreads(96, 1);
    dim3 sGrid(25, 1);
    
    dim3 nThreads(128, 1);
    dim3 nGrid(79, 1);
    
    calcSynapses <<< sGrid, sThreads >>> (t);
    calcNeurons <<< nGrid, nThreads >>> (t);
    iT++;
    t= iT*DT;
}

