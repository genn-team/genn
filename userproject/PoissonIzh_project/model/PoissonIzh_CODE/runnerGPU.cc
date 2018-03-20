
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model PoissonIzh containing the host side code for a GPU simulator version.
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

void pushIzh1StateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VIzh1, VIzh1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UIzh1, UIzh1, 10 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushIzh1SpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntIzh1, glbSpkCntIzh1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkIzh1, glbSpkIzh1, 10 * sizeof(unsigned int), cudaMemcpyHostToDevice));
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

void pushPNStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_timeStepToSpikePN, timeStepToSpikePN, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushPNSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPN, glbSpkCntPN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPN, glbSpkPN, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushPNSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushPNCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPN, glbSpkCntPN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPN, glbSpkPN, glbSpkCntPN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushPNCurrentSpikeEventsToDevice()
 {
}

void pushPNIzh1StateToDevice(bool hostInitialisedOnly)
 {
    const size_t size = 1000;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gPNIzh1, gPNIzh1, size * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynPNIzh1, inSynPNIzh1, 10 * sizeof(float), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------
// copying things from device

void pullIzh1StateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VIzh1, d_VIzh1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(UIzh1, d_UIzh1, 10 * sizeof(scalar), cudaMemcpyDeviceToHost));
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

void pullPNStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(timeStepToSpikePN, d_timeStepToSpikePN, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullPNSpikeEventsFromDevice()
 {
}

void pullPNSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPN, d_glbSpkCntPN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPN, d_glbSpkPN, glbSpkCntPN [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullPNSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullPNCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPN, d_glbSpkCntPN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPN, d_glbSpkPN, glbSpkCntPN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullPNCurrentSpikeEventsFromDevice()
 {
}

void pullPNIzh1StateFromDevice()
 {
    size_t size = 1000;
    CHECK_CUDA_ERRORS(cudaMemcpy(gPNIzh1, d_gPNIzh1, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynPNIzh1, d_inSynPNIzh1, 10 * sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice(bool hostInitialisedOnly)
 {
    pushIzh1StateToDevice(hostInitialisedOnly);
    pushIzh1SpikesToDevice(hostInitialisedOnly);
    pushPNStateToDevice(hostInitialisedOnly);
    pushPNSpikesToDevice(hostInitialisedOnly);
    pushPNIzh1StateToDevice(hostInitialisedOnly);
}

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushIzh1SpikesToDevice();
    pushPNSpikesToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushIzh1CurrentSpikesToDevice();
    pushPNCurrentSpikesToDevice();
}
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushIzh1SpikeEventsToDevice();
    pushPNSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushIzh1CurrentSpikeEventsToDevice();
    pushPNCurrentSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullIzh1StateFromDevice();
    pullIzh1SpikesFromDevice();
    pullPNStateFromDevice();
    pullPNSpikesFromDevice();
    pullPNIzh1StateFromDevice();
}

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
    pullIzh1SpikesFromDevice();
    pullPNSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
    pullIzh1CurrentSpikesFromDevice();
    pullPNCurrentSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntIzh1, d_glbSpkCntIzh1, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPN, d_glbSpkCntPN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}


// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
    pullIzh1SpikeEventsFromDevice();
    pullPNSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
    pullIzh1CurrentSpikeEventsFromDevice();
    pullPNCurrentSpikeEventsFromDevice();
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
    
    //model.padSumSynapseTrgN[model.synapseGrpN - 1] is 32
    dim3 sThreads(32, 1);
    dim3 sGrid(1, 1);
    
    dim3 nThreads(32, 1);
    dim3 nGrid(5, 1);
    
    calcSynapses <<< sGrid, sThreads >>> (t);
    calcNeurons <<< nGrid, nThreads >>> (t);
    iT++;
    t= iT*DT;
}

