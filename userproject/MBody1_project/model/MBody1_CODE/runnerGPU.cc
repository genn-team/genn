
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model MBody1 containing the host side code for a GPU simulator version.
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

void pushDNStateToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VDN, VDN, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mDN, mDN, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_hDN, hDN, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_nDN, nDN, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushDNSpikesToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntDN, glbSpkCntDN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkDN, glbSpkDN, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    pushDNSpikeEventsToDevice(hostInitialisedOnly);
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_sTDN, sTDN, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushDNSpikeEventsToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntDN, glbSpkCntEvntDN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntDN, glbSpkEvntDN, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushDNCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntDN, glbSpkCntDN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkDN, glbSpkDN, glbSpkCntDN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushDNCurrentSpikeEventsToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntDN, glbSpkCntEvntDN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntDN, glbSpkEvntDN, glbSpkCntEvntDN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushKCStateToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VKC, VKC, 1000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mKC, mKC, 1000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_hKC, hKC, 1000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_nKC, nKC, 1000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushKCSpikesToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntKC, glbSpkCntKC, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkKC, glbSpkKC, 1000 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_sTKC, sTKC, 1000 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushKCSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushKCCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntKC, glbSpkCntKC, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkKC, glbSpkKC, glbSpkCntKC[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushKCCurrentSpikeEventsToDevice()
 {
}

void pushLHIStateToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VLHI, VLHI, 20 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_mLHI, mLHI, 20 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_hLHI, hLHI, 20 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_nLHI, nLHI, 20 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushLHISpikesToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntLHI, glbSpkCntLHI, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkLHI, glbSpkLHI, 20 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    pushLHISpikeEventsToDevice(hostInitialisedOnly);
}

void pushLHISpikeEventsToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntLHI, glbSpkCntEvntLHI, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntLHI, glbSpkEvntLHI, 20 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushLHICurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntLHI, glbSpkCntLHI, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkLHI, glbSpkLHI, glbSpkCntLHI[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushLHICurrentSpikeEventsToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntLHI, glbSpkCntEvntLHI, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntLHI, glbSpkEvntLHI, glbSpkCntEvntLHI[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushPNStateToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_VPN, VPN, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_seedPN, seedPN, 100 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_spikeTimePN, spikeTimePN, 100 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
}

void pushPNSpikesToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPN, glbSpkCntPN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPN, glbSpkPN, 100 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
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

void pushDNDNStateToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynDNDN, inSynDNDN, 100 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushKCDNStateToDevice(bool hostInitialisedOnly)
 {
    const size_t size = 100000;
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gKCDN, gKCDN, size * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gRawKCDN, gRawKCDN, size * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynKCDN, inSynKCDN, 100 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushLHIKCStateToDevice(bool hostInitialisedOnly)
 {
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynLHIKC, inSynLHIKC, 1000 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushPNKCStateToDevice(bool hostInitialisedOnly)
 {
    const size_t size = 100000;
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gPNKC, gPNKC, size * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynPNKC, inSynPNKC, 1000 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void pushPNLHIStateToDevice(bool hostInitialisedOnly)
 {
    const size_t size = 2000;
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gPNLHI, gPNLHI, size * sizeof(scalar), cudaMemcpyHostToDevice));
    }
    if(!hostInitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynPNLHI, inSynPNLHI, 20 * sizeof(float), cudaMemcpyHostToDevice));
    }
}

// ------------------------------------------------------------------------
// copying things from device

void pullDNStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VDN, d_VDN, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(mDN, d_mDN, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(hDN, d_hDN, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(nDN, d_nDN, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullDNSpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntDN, d_glbSpkCntEvntDN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntDN, d_glbSpkEvntDN, 100 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullDNSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntDN, d_glbSpkCntDN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkDN, d_glbSpkDN, glbSpkCntDN [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    pullDNSpikeEventsFromDevice();
}

void pullDNSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    CHECK_CUDA_ERRORS(cudaMemcpy(sTDN, d_sTDN, glbSpkCntDN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullDNCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntDN, d_glbSpkCntDN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkDN, d_glbSpkDN, glbSpkCntDN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullDNCurrentSpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntDN, d_glbSpkCntEvntDN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntDN, d_glbSpkEvntDN, glbSpkCntEvntDN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullKCStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VKC, d_VKC, 1000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(mKC, d_mKC, 1000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(hKC, d_hKC, 1000 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(nKC, d_nKC, 1000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullKCSpikeEventsFromDevice()
 {
}

void pullKCSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntKC, d_glbSpkCntKC, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkKC, d_glbSpkKC, glbSpkCntKC [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullKCSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    CHECK_CUDA_ERRORS(cudaMemcpy(sTKC, d_sTKC, glbSpkCntKC[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullKCCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntKC, d_glbSpkCntKC, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkKC, d_glbSpkKC, glbSpkCntKC[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullKCCurrentSpikeEventsFromDevice()
 {
}

void pullLHIStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VLHI, d_VLHI, 20 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(mLHI, d_mLHI, 20 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(hLHI, d_hLHI, 20 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(nLHI, d_nLHI, 20 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullLHISpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntLHI, d_glbSpkCntEvntLHI, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntLHI, d_glbSpkEvntLHI, 20 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullLHISpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntLHI, d_glbSpkCntLHI, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkLHI, d_glbSpkLHI, glbSpkCntLHI [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    pullLHISpikeEventsFromDevice();
}

void pullLHISpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullLHICurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntLHI, d_glbSpkCntLHI, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkLHI, d_glbSpkLHI, glbSpkCntLHI[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullLHICurrentSpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntLHI, d_glbSpkCntEvntLHI, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntLHI, d_glbSpkEvntLHI, glbSpkCntEvntLHI[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullPNStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VPN, d_VPN, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(seedPN, d_seedPN, 100 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(spikeTimePN, d_spikeTimePN, 100 * sizeof(scalar), cudaMemcpyDeviceToHost));
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

void pullDNDNStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynDNDN, d_inSynDNDN, 100 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullKCDNStateFromDevice()
 {
    size_t size = 100000;
    CHECK_CUDA_ERRORS(cudaMemcpy(gKCDN, d_gKCDN, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(gRawKCDN, d_gRawKCDN, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynKCDN, d_inSynKCDN, 100 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullLHIKCStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynLHIKC, d_inSynLHIKC, 1000 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullPNKCStateFromDevice()
 {
    size_t size = 100000;
    CHECK_CUDA_ERRORS(cudaMemcpy(gPNKC, d_gPNKC, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynPNKC, d_inSynPNKC, 1000 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullPNLHIStateFromDevice()
 {
    size_t size = 2000;
    CHECK_CUDA_ERRORS(cudaMemcpy(gPNLHI, d_gPNLHI, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynPNLHI, d_inSynPNLHI, 20 * sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice(bool hostInitialisedOnly)
 {
    pushDNStateToDevice(hostInitialisedOnly);
    pushDNSpikesToDevice(hostInitialisedOnly);
    pushKCStateToDevice(hostInitialisedOnly);
    pushKCSpikesToDevice(hostInitialisedOnly);
    pushLHIStateToDevice(hostInitialisedOnly);
    pushLHISpikesToDevice(hostInitialisedOnly);
    pushPNStateToDevice(hostInitialisedOnly);
    pushPNSpikesToDevice(hostInitialisedOnly);
    pushDNDNStateToDevice(hostInitialisedOnly);
    pushKCDNStateToDevice(hostInitialisedOnly);
    pushLHIKCStateToDevice(hostInitialisedOnly);
    pushPNKCStateToDevice(hostInitialisedOnly);
    pushPNLHIStateToDevice(hostInitialisedOnly);
}

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushDNSpikesToDevice();
    pushKCSpikesToDevice();
    pushLHISpikesToDevice();
    pushPNSpikesToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushDNCurrentSpikesToDevice();
    pushKCCurrentSpikesToDevice();
    pushLHICurrentSpikesToDevice();
    pushPNCurrentSpikesToDevice();
}
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushDNSpikeEventsToDevice();
    pushKCSpikeEventsToDevice();
    pushLHISpikeEventsToDevice();
    pushPNSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushDNCurrentSpikeEventsToDevice();
    pushKCCurrentSpikeEventsToDevice();
    pushLHICurrentSpikeEventsToDevice();
    pushPNCurrentSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullDNStateFromDevice();
    pullDNSpikesFromDevice();
    pullKCStateFromDevice();
    pullKCSpikesFromDevice();
    pullLHIStateFromDevice();
    pullLHISpikesFromDevice();
    pullPNStateFromDevice();
    pullPNSpikesFromDevice();
    pullDNDNStateFromDevice();
    pullKCDNStateFromDevice();
    pullLHIKCStateFromDevice();
    pullPNKCStateFromDevice();
    pullPNLHIStateFromDevice();
}

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
    pullDNSpikesFromDevice();
    pullKCSpikesFromDevice();
    pullLHISpikesFromDevice();
    pullPNSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
    pullDNCurrentSpikesFromDevice();
    pullKCCurrentSpikesFromDevice();
    pullLHICurrentSpikesFromDevice();
    pullPNCurrentSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntDN, d_glbSpkCntDN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntKC, d_glbSpkCntKC, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntLHI, d_glbSpkCntLHI, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPN, d_glbSpkCntPN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}


// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
    pullDNSpikeEventsFromDevice();
    pullKCSpikeEventsFromDevice();
    pullLHISpikeEventsFromDevice();
    pullPNSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
    pullDNCurrentSpikeEventsFromDevice();
    pullKCCurrentSpikeEventsFromDevice();
    pullLHICurrentSpikeEventsFromDevice();
    pullPNCurrentSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void copySpikeEventNFromDevice()
 {
    
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntDN, d_glbSpkCntEvntDN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntLHI, d_glbSpkCntEvntLHI, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}


// ------------------------------------------------------------------------
// the time stepping procedure (using GPU)
void stepTimeGPU()
 {
    
    //model.padSumSynapseTrgN[model.synapseGrpN - 1] is 2592
    dim3 sThreads(96, 1);
    dim3 sGrid(27, 1);
    
    dim3 lThreads(32, 1);
    dim3 lGrid(32, 1);
    
    dim3 nThreads(64, 1);
    dim3 nGrid(21, 1);
    
    calcSynapses <<< sGrid, sThreads >>> (t);
    learnSynapsesPost <<< lGrid, lThreads >>> (t);
    calcNeurons <<< nGrid, nThreads >>> (offsetPN, ratesPN, t);
    iT++;
    t= iT*DT;
}

