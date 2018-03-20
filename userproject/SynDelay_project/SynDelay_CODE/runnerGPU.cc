
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model SynDelay containing the host side code for a GPU simulator version.
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

void pushInputStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VInput, VInput, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UInput, UInput, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushInputSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntInput, glbSpkCntInput, 7 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkInput, glbSpkInput, 3500 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushInputSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushInputCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntInput+spkQuePtrInput, glbSpkCntInput+spkQuePtrInput, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkInput+(spkQuePtrInput*500), glbSpkInput+(spkQuePtrInput*500), glbSpkCntInput[spkQuePtrInput] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushInputCurrentSpikeEventsToDevice()
 {
}

void pushInterStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VInter, VInter, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UInter, UInter, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushInterSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntInter, glbSpkCntInter, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkInter, glbSpkInter, 500 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushInterSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushInterCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntInter, glbSpkCntInter, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkInter, glbSpkInter, glbSpkCntInter[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushInterCurrentSpikeEventsToDevice()
 {
}

void pushOutputStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VOutput, VOutput, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_UOutput, UOutput, 500 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushOutputSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntOutput, glbSpkCntOutput, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkOutput, glbSpkOutput, 500 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushOutputSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushOutputCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntOutput, glbSpkCntOutput, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkOutput, glbSpkOutput, glbSpkCntOutput[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushOutputCurrentSpikeEventsToDevice()
 {
}

void pushInputInterStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynInputInter, inSynInputInter, 500 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushInputOutputStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynInputOutput, inSynInputOutput, 500 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushInterOutputStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynInterOutput, inSynInterOutput, 500 * sizeof(float), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------
// copying things from device

void pullInputStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VInput, d_VInput, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(UInput, d_UInput, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullInputSpikeEventsFromDevice()
 {
}

void pullInputSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInput, d_glbSpkCntInput, 7 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkInput, d_glbSpkInput, glbSpkCntInput [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullInputSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullInputCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInput+spkQuePtrInput, d_glbSpkCntInput+spkQuePtrInput, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkInput+(spkQuePtrInput*500), d_glbSpkInput+(spkQuePtrInput*500), glbSpkCntInput[spkQuePtrInput] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullInputCurrentSpikeEventsFromDevice()
 {
}

void pullInterStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VInter, d_VInter, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(UInter, d_UInter, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullInterSpikeEventsFromDevice()
 {
}

void pullInterSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInter, d_glbSpkCntInter, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkInter, d_glbSpkInter, glbSpkCntInter [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullInterSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullInterCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInter, d_glbSpkCntInter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkInter, d_glbSpkInter, glbSpkCntInter[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullInterCurrentSpikeEventsFromDevice()
 {
}

void pullOutputStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VOutput, d_VOutput, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(UOutput, d_UOutput, 500 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullOutputSpikeEventsFromDevice()
 {
}

void pullOutputSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntOutput, d_glbSpkCntOutput, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkOutput, d_glbSpkOutput, glbSpkCntOutput [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullOutputSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullOutputCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntOutput, d_glbSpkCntOutput, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkOutput, d_glbSpkOutput, glbSpkCntOutput[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullOutputCurrentSpikeEventsFromDevice()
 {
}

void pullInputInterStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynInputInter, d_inSynInputInter, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullInputOutputStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynInputOutput, d_inSynInputOutput, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullInterOutputStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynInterOutput, d_inSynInterOutput, 500 * sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice(bool hostInitialisedOnly)
 {
    pushInputStateToDevice(hostInitialisedOnly);
    pushInputSpikesToDevice(hostInitialisedOnly);
    pushInterStateToDevice(hostInitialisedOnly);
    pushInterSpikesToDevice(hostInitialisedOnly);
    pushOutputStateToDevice(hostInitialisedOnly);
    pushOutputSpikesToDevice(hostInitialisedOnly);
    pushInputInterStateToDevice(hostInitialisedOnly);
    pushInputOutputStateToDevice(hostInitialisedOnly);
    pushInterOutputStateToDevice(hostInitialisedOnly);
}

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushInputSpikesToDevice();
    pushInterSpikesToDevice();
    pushOutputSpikesToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushInputCurrentSpikesToDevice();
    pushInterCurrentSpikesToDevice();
    pushOutputCurrentSpikesToDevice();
}
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushInputSpikeEventsToDevice();
    pushInterSpikeEventsToDevice();
    pushOutputSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushInputCurrentSpikeEventsToDevice();
    pushInterCurrentSpikeEventsToDevice();
    pushOutputCurrentSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullInputStateFromDevice();
    pullInputSpikesFromDevice();
    pullInterStateFromDevice();
    pullInterSpikesFromDevice();
    pullOutputStateFromDevice();
    pullOutputSpikesFromDevice();
    pullInputInterStateFromDevice();
    pullInputOutputStateFromDevice();
    pullInterOutputStateFromDevice();
}

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
    pullInputSpikesFromDevice();
    pullInterSpikesFromDevice();
    pullOutputSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
    pullInputCurrentSpikesFromDevice();
    pullInterCurrentSpikesFromDevice();
    pullOutputCurrentSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInput, d_glbSpkCntInput, 7* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInter, d_glbSpkCntInter, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntOutput, d_glbSpkCntOutput, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}


// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
    pullInputSpikeEventsFromDevice();
    pullInterSpikeEventsFromDevice();
    pullOutputSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
    pullInputCurrentSpikeEventsFromDevice();
    pullInterCurrentSpikeEventsFromDevice();
    pullOutputCurrentSpikeEventsFromDevice();
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
    
    //model.padSumSynapseTrgN[model.synapseGrpN - 1] is 1536
    dim3 sThreads(64, 1);
    dim3 sGrid(24, 1);
    
    dim3 nThreads(64, 1);
    dim3 nGrid(24, 1);
    
    calcSynapses <<< sGrid, sThreads >>> (t);
    spkQuePtrInput = (spkQuePtrInput + 1) % 7;
    calcNeurons <<< nGrid, nThreads >>> (t);
    iT++;
    t= iT*DT;
}

