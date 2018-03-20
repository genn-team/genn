
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model model containing the host side code for a GPU simulator version.
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

void pushCortexStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aCortex, aCortex, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inCortex, inCortex, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_outCortex, outCortex, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCortexSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntCortex, glbSpkCntCortex, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCortex, glbSpkCortex, 6 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushCortexSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushCortexCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntCortex, glbSpkCntCortex, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCortex, glbSpkCortex, glbSpkCntCortex[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushCortexCurrentSpikeEventsToDevice()
 {
}

void pushD1StateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aD1, aD1, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_outD1, outD1, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushD1SpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntD1, glbSpkCntD1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkD1, glbSpkD1, 6 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushD1SpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushD1CurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntD1, glbSpkCntD1, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkD1, glbSpkD1, glbSpkCntD1[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushD1CurrentSpikeEventsToDevice()
 {
}

void pushD2StateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aD2, aD2, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_outD2, outD2, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushD2SpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntD2, glbSpkCntD2, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkD2, glbSpkD2, 6 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushD2SpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushD2CurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntD2, glbSpkCntD2, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkD2, glbSpkD2, glbSpkCntD2[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushD2CurrentSpikeEventsToDevice()
 {
}

void pushGPeStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aGPe, aGPe, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_outGPe, outGPe, 12 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushGPeSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntGPe, glbSpkCntGPe, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkGPe, glbSpkGPe, 6 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushGPeSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushGPeCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntGPe, glbSpkCntGPe, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkGPe, glbSpkGPe, glbSpkCntGPe[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushGPeCurrentSpikeEventsToDevice()
 {
}

void pushGPiStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aGPi, aGPi, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_outGPi, outGPi, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushGPiSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntGPi, glbSpkCntGPi, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkGPi, glbSpkGPi, 6 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushGPiSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushGPiCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntGPi, glbSpkCntGPi, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkGPi, glbSpkGPi, glbSpkCntGPi[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushGPiCurrentSpikeEventsToDevice()
 {
}

void pushSTNStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aSTN, aSTN, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_outSTN, outSTN, 6 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushSTNSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntSTN, glbSpkCntSTN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkSTN, glbSpkSTN, 6 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushSTNSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushSTNCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntSTN, glbSpkCntSTN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkSTN, glbSpkSTN, glbSpkCntSTN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushSTNCurrentSpikeEventsToDevice()
 {
}

void pushCortex_to_D1_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynCortex_to_D1_Synapse_0_weight_update, inSynCortex_to_D1_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushCortex_to_D2_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynCortex_to_D2_Synapse_0_weight_update, inSynCortex_to_D2_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushCortex_to_STN_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynCortex_to_STN_Synapse_0_weight_update, inSynCortex_to_STN_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushD1_to_GPi_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynD1_to_GPi_Synapse_0_weight_update, inSynD1_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushD2_to_GPe_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynD2_to_GPe_Synapse_0_weight_update, inSynD2_to_GPe_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushGPe_to_GPi_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynGPe_to_GPi_Synapse_0_weight_update, inSynGPe_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushGPe_to_STN_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynGPe_to_STN_Synapse_0_weight_update, inSynGPe_to_STN_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushSTN_to_GPe_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynSTN_to_GPe_Synapse_0_weight_update, inSynSTN_to_GPe_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushSTN_to_GPi_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynSTN_to_GPi_Synapse_0_weight_update, inSynSTN_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------
// copying things from device

void pullCortexStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(aCortex, d_aCortex, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inCortex, d_inCortex, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(outCortex, d_outCortex, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCortexSpikeEventsFromDevice()
 {
}

void pullCortexSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntCortex, d_glbSpkCntCortex, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCortex, d_glbSpkCortex, glbSpkCntCortex [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullCortexSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullCortexCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntCortex, d_glbSpkCntCortex, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCortex, d_glbSpkCortex, glbSpkCntCortex[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullCortexCurrentSpikeEventsFromDevice()
 {
}

void pullD1StateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(aD1, d_aD1, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(outD1, d_outD1, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullD1SpikeEventsFromDevice()
 {
}

void pullD1SpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntD1, d_glbSpkCntD1, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkD1, d_glbSpkD1, glbSpkCntD1 [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullD1SpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullD1CurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntD1, d_glbSpkCntD1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkD1, d_glbSpkD1, glbSpkCntD1[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullD1CurrentSpikeEventsFromDevice()
 {
}

void pullD2StateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(aD2, d_aD2, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(outD2, d_outD2, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullD2SpikeEventsFromDevice()
 {
}

void pullD2SpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntD2, d_glbSpkCntD2, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkD2, d_glbSpkD2, glbSpkCntD2 [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullD2SpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullD2CurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntD2, d_glbSpkCntD2, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkD2, d_glbSpkD2, glbSpkCntD2[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullD2CurrentSpikeEventsFromDevice()
 {
}

void pullGPeStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(aGPe, d_aGPe, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(outGPe, d_outGPe, 12 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullGPeSpikeEventsFromDevice()
 {
}

void pullGPeSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntGPe, d_glbSpkCntGPe, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkGPe, d_glbSpkGPe, glbSpkCntGPe [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullGPeSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullGPeCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntGPe, d_glbSpkCntGPe, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkGPe, d_glbSpkGPe, glbSpkCntGPe[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullGPeCurrentSpikeEventsFromDevice()
 {
}

void pullGPiStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(aGPi, d_aGPi, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(outGPi, d_outGPi, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullGPiSpikeEventsFromDevice()
 {
}

void pullGPiSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntGPi, d_glbSpkCntGPi, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkGPi, d_glbSpkGPi, glbSpkCntGPi [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullGPiSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullGPiCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntGPi, d_glbSpkCntGPi, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkGPi, d_glbSpkGPi, glbSpkCntGPi[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullGPiCurrentSpikeEventsFromDevice()
 {
}

void pullSTNStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(aSTN, d_aSTN, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(outSTN, d_outSTN, 6 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullSTNSpikeEventsFromDevice()
 {
}

void pullSTNSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntSTN, d_glbSpkCntSTN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkSTN, d_glbSpkSTN, glbSpkCntSTN [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullSTNSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullSTNCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntSTN, d_glbSpkCntSTN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkSTN, d_glbSpkSTN, glbSpkCntSTN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullSTNCurrentSpikeEventsFromDevice()
 {
}

void pullCortex_to_D1_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynCortex_to_D1_Synapse_0_weight_update, d_inSynCortex_to_D1_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullCortex_to_D2_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynCortex_to_D2_Synapse_0_weight_update, d_inSynCortex_to_D2_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullCortex_to_STN_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynCortex_to_STN_Synapse_0_weight_update, d_inSynCortex_to_STN_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullD1_to_GPi_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynD1_to_GPi_Synapse_0_weight_update, d_inSynD1_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullD2_to_GPe_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynD2_to_GPe_Synapse_0_weight_update, d_inSynD2_to_GPe_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullGPe_to_GPi_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynGPe_to_GPi_Synapse_0_weight_update, d_inSynGPe_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullGPe_to_STN_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynGPe_to_STN_Synapse_0_weight_update, d_inSynGPe_to_STN_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullSTN_to_GPe_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynSTN_to_GPe_Synapse_0_weight_update, d_inSynSTN_to_GPe_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullSTN_to_GPi_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynSTN_to_GPi_Synapse_0_weight_update, d_inSynSTN_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice(bool hostInitialisedOnly)
 {
    pushCortexStateToDevice(hostInitialisedOnly);
    pushCortexSpikesToDevice(hostInitialisedOnly);
    pushD1StateToDevice(hostInitialisedOnly);
    pushD1SpikesToDevice(hostInitialisedOnly);
    pushD2StateToDevice(hostInitialisedOnly);
    pushD2SpikesToDevice(hostInitialisedOnly);
    pushGPeStateToDevice(hostInitialisedOnly);
    pushGPeSpikesToDevice(hostInitialisedOnly);
    pushGPiStateToDevice(hostInitialisedOnly);
    pushGPiSpikesToDevice(hostInitialisedOnly);
    pushSTNStateToDevice(hostInitialisedOnly);
    pushSTNSpikesToDevice(hostInitialisedOnly);
    pushCortex_to_D1_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushCortex_to_D2_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushCortex_to_STN_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushD1_to_GPi_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushD2_to_GPe_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushGPe_to_GPi_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushGPe_to_STN_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushSTN_to_GPe_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushSTN_to_GPi_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
}

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushCortexSpikesToDevice();
    pushD1SpikesToDevice();
    pushD2SpikesToDevice();
    pushGPeSpikesToDevice();
    pushGPiSpikesToDevice();
    pushSTNSpikesToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushCortexCurrentSpikesToDevice();
    pushD1CurrentSpikesToDevice();
    pushD2CurrentSpikesToDevice();
    pushGPeCurrentSpikesToDevice();
    pushGPiCurrentSpikesToDevice();
    pushSTNCurrentSpikesToDevice();
}
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushCortexSpikeEventsToDevice();
    pushD1SpikeEventsToDevice();
    pushD2SpikeEventsToDevice();
    pushGPeSpikeEventsToDevice();
    pushGPiSpikeEventsToDevice();
    pushSTNSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushCortexCurrentSpikeEventsToDevice();
    pushD1CurrentSpikeEventsToDevice();
    pushD2CurrentSpikeEventsToDevice();
    pushGPeCurrentSpikeEventsToDevice();
    pushGPiCurrentSpikeEventsToDevice();
    pushSTNCurrentSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullCortexStateFromDevice();
    pullCortexSpikesFromDevice();
    pullD1StateFromDevice();
    pullD1SpikesFromDevice();
    pullD2StateFromDevice();
    pullD2SpikesFromDevice();
    pullGPeStateFromDevice();
    pullGPeSpikesFromDevice();
    pullGPiStateFromDevice();
    pullGPiSpikesFromDevice();
    pullSTNStateFromDevice();
    pullSTNSpikesFromDevice();
    pullCortex_to_D1_Synapse_0_weight_updateStateFromDevice();
    pullCortex_to_D2_Synapse_0_weight_updateStateFromDevice();
    pullCortex_to_STN_Synapse_0_weight_updateStateFromDevice();
    pullD1_to_GPi_Synapse_0_weight_updateStateFromDevice();
    pullD2_to_GPe_Synapse_0_weight_updateStateFromDevice();
    pullGPe_to_GPi_Synapse_0_weight_updateStateFromDevice();
    pullGPe_to_STN_Synapse_0_weight_updateStateFromDevice();
    pullSTN_to_GPe_Synapse_0_weight_updateStateFromDevice();
    pullSTN_to_GPi_Synapse_0_weight_updateStateFromDevice();
}

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
    pullCortexSpikesFromDevice();
    pullD1SpikesFromDevice();
    pullD2SpikesFromDevice();
    pullGPeSpikesFromDevice();
    pullGPiSpikesFromDevice();
    pullSTNSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
    pullCortexCurrentSpikesFromDevice();
    pullD1CurrentSpikesFromDevice();
    pullD2CurrentSpikesFromDevice();
    pullGPeCurrentSpikesFromDevice();
    pullGPiCurrentSpikesFromDevice();
    pullSTNCurrentSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntCortex, d_glbSpkCntCortex, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntD1, d_glbSpkCntD1, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntD2, d_glbSpkCntD2, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntGPe, d_glbSpkCntGPe, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntGPi, d_glbSpkCntGPi, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntSTN, d_glbSpkCntSTN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}


// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
    pullCortexSpikeEventsFromDevice();
    pullD1SpikeEventsFromDevice();
    pullD2SpikeEventsFromDevice();
    pullGPeSpikeEventsFromDevice();
    pullGPiSpikeEventsFromDevice();
    pullSTNSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
    pullCortexCurrentSpikeEventsFromDevice();
    pullD1CurrentSpikeEventsFromDevice();
    pullD2CurrentSpikeEventsFromDevice();
    pullGPeCurrentSpikeEventsFromDevice();
    pullGPiCurrentSpikeEventsFromDevice();
    pullSTNCurrentSpikeEventsFromDevice();
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
    
    //model.padSumSynapseTrgN[model.synapseGrpN - 1] is 288
    dim3 sThreads(32, 1);
    dim3 sGrid(9, 1);
    
    dim3 sDThreads(32, 1);
    dim3 sDGrid(11, 1);
    
    dim3 nThreads(32, 1);
    dim3 nGrid(6, 1);
    
    calcSynapseDynamics <<< sDGrid, sDThreads >>> (t);
    calcSynapses <<< sGrid, sThreads >>> (t);
    spkQuePtrGPe = (spkQuePtrGPe + 1) % 2;
    calcNeurons <<< nGrid, nThreads >>> (t);
    iT++;
    t= iT*DT;
}

