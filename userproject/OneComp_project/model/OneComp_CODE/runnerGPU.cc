
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model OneComp containing the host side code for a GPU simulator version.
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
// ------------------------------------------------------------------------
// copying things to device

void pushIzh1StateToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VIzh1, VIzh1, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_UIzh1, UIzh1, 1 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushIzh1SpikesToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntIzh1, glbSpkCntIzh1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkIzh1, glbSpkIzh1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushIzh1SpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushIzh1CurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntIzh1, glbSpkCntIzh1, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkIzh1, glbSpkIzh1, glbSpkCntIzh1[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushIzh1CurrentSpikeEventsToDevice()
 {
}

// ------------------------------------------------------------------------
// copying things from device

void pullIzh1StateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VIzh1, d_VIzh1, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(UIzh1, d_UIzh1, 1 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullIzh1SpikeEventsFromDevice()
 {
}

void pullIzh1SpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntIzh1, d_glbSpkCntIzh1, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkIzh1, d_glbSpkIzh1, glbSpkCntIzh1 [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullIzh1SpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullIzh1CurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntIzh1, d_glbSpkCntIzh1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkIzh1, d_glbSpkIzh1, glbSpkCntIzh1[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullIzh1CurrentSpikeEventsFromDevice()
 {
}

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice(bool hostInitialisedOnly)
 {
    pushIzh1StateToDevice(hostInitialisedOnly);
    pushIzh1SpikesToDevice(hostInitialisedOnly);
}

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushIzh1SpikesToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushIzh1CurrentSpikesToDevice();
}
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushIzh1SpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushIzh1CurrentSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullIzh1StateFromDevice();
    pullIzh1SpikesFromDevice();
}

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
    pullIzh1SpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
    pullIzh1CurrentSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntIzh1, d_glbSpkCntIzh1, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}


// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
    pullIzh1SpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
    pullIzh1CurrentSpikeEventsFromDevice();
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
    
    dim3 nThreads(32, 1);
    dim3 nGrid(1, 1);
    
    calcNeurons <<< nGrid, nThreads >>> (t);
    iT++;
    t= iT*DT;
}

