
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model Schmuker_2014_classifier containing the host side code for a GPU simulator version.
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

#include "neuronKrnl.cc"
#include "synapseKrnl.cc"
// ------------------------------------------------------------------------
// copying things to device

void pushANStateToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VAN, VAN, 180 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_preVAN, preVAN, 180 * sizeof(scalar), cudaMemcpyHostToDevice));
    }

void pushANSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntAN, glbSpkCntAN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkAN, glbSpkAN, 180 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    pushANSpikeEventsToDevice();
    }

void pushANSpikeEventsToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntAN, glbSpkCntEvntAN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntAN, glbSpkEvntAN, 180 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushANCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntAN, glbSpkCntAN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkAN, glbSpkAN, glbSpkCntAN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushANCurrentSpikeEventsToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntAN, glbSpkCntEvntAN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntAN, glbSpkEvntAN, glbSpkCntEvntAN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushPNStateToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VPN, VPN, 600 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_preVPN, preVPN, 600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }

void pushPNSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPN, glbSpkCntPN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPN, glbSpkPN, 600 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    pushPNSpikeEventsToDevice();
    }

void pushPNSpikeEventsToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntPN, glbSpkCntEvntPN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntPN, glbSpkEvntPN, 600 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushPNCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntPN, glbSpkCntPN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkPN, glbSpkPN, glbSpkCntPN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushPNCurrentSpikeEventsToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntPN, glbSpkCntEvntPN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntPN, glbSpkEvntPN, glbSpkCntEvntPN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushRNStateToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VRN, VRN, 600 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_seedRN, seedRN, 600 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_spikeTimeRN, spikeTimeRN, 600 * sizeof(scalar), cudaMemcpyHostToDevice));
    }

void pushRNSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntRN, glbSpkCntRN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkRN, glbSpkRN, 600 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    pushRNSpikeEventsToDevice();
    }

void pushRNSpikeEventsToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntRN, glbSpkCntEvntRN, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntRN, glbSpkEvntRN, 600 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushRNCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntRN, glbSpkCntRN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkRN, glbSpkRN, glbSpkCntRN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushRNCurrentSpikeEventsToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntEvntRN, glbSpkCntEvntRN, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkEvntRN, glbSpkEvntRN, glbSpkCntEvntRN[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushANANStateToDevice()
 {
    size_t size = 32400;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gANAN, gANAN, size * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynANAN, inSynANAN, 180 * sizeof(float), cudaMemcpyHostToDevice));
    }

void pushPNANStateToDevice()
 {
    size_t size = 108000;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gPNAN, gPNAN, size * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynPNAN, inSynPNAN, 180 * sizeof(float), cudaMemcpyHostToDevice));
    }

void pushPNPNStateToDevice()
 {
    size_t size = 360000;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gPNPN, gPNPN, size * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynPNPN, inSynPNPN, 600 * sizeof(float), cudaMemcpyHostToDevice));
    }

void pushRNPNStateToDevice()
 {
    size_t size = 360000;
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gRNPN, gRNPN, size * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynRNPN, inSynRNPN, 600 * sizeof(float), cudaMemcpyHostToDevice));
    }

// ------------------------------------------------------------------------
// copying things from device

void pullANStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VAN, d_VAN, 180 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(preVAN, d_preVAN, 180 * sizeof(scalar), cudaMemcpyDeviceToHost));
    }

void pullANSpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntAN, d_glbSpkCntEvntAN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntAN, d_glbSpkEvntAN, 180 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullANSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntAN, d_glbSpkCntAN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkAN, d_glbSpkAN, glbSpkCntAN [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    pullANSpikeEventsFromDevice();
    }

void pullANSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullANCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntAN, d_glbSpkCntAN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkAN, d_glbSpkAN, glbSpkCntAN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullANCurrentSpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntAN, d_glbSpkCntEvntAN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntAN, d_glbSpkEvntAN, glbSpkCntEvntAN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullPNStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VPN, d_VPN, 600 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(preVPN, d_preVPN, 600 * sizeof(scalar), cudaMemcpyDeviceToHost));
    }

void pullPNSpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntPN, d_glbSpkCntEvntPN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntPN, d_glbSpkEvntPN, 600 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullPNSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPN, d_glbSpkCntPN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkPN, d_glbSpkPN, glbSpkCntPN [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    pullPNSpikeEventsFromDevice();
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
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntPN, d_glbSpkCntEvntPN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntPN, d_glbSpkEvntPN, glbSpkCntEvntPN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullRNStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VRN, d_VRN, 600 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(seedRN, d_seedRN, 600 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(spikeTimeRN, d_spikeTimeRN, 600 * sizeof(scalar), cudaMemcpyDeviceToHost));
    }

void pullRNSpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntRN, d_glbSpkCntEvntRN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntRN, d_glbSpkEvntRN, 600 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullRNSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntRN, d_glbSpkCntRN, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkRN, d_glbSpkRN, glbSpkCntRN [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    pullRNSpikeEventsFromDevice();
    }

void pullRNSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullRNCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntRN, d_glbSpkCntRN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkRN, d_glbSpkRN, glbSpkCntRN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullRNCurrentSpikeEventsFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntRN, d_glbSpkCntEvntRN, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkEvntRN, d_glbSpkEvntRN, glbSpkCntEvntRN[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullANANStateFromDevice()
 {
    size_t size = 32400;
    CHECK_CUDA_ERRORS(cudaMemcpy(gANAN, d_gANAN, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynANAN, d_inSynANAN, 180 * sizeof(float), cudaMemcpyDeviceToHost));
    }

void pullPNANStateFromDevice()
 {
    size_t size = 108000;
    CHECK_CUDA_ERRORS(cudaMemcpy(gPNAN, d_gPNAN, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynPNAN, d_inSynPNAN, 180 * sizeof(float), cudaMemcpyDeviceToHost));
    }

void pullPNPNStateFromDevice()
 {
    size_t size = 360000;
    CHECK_CUDA_ERRORS(cudaMemcpy(gPNPN, d_gPNPN, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynPNPN, d_inSynPNPN, 600 * sizeof(float), cudaMemcpyDeviceToHost));
    }

void pullRNPNStateFromDevice()
 {
    size_t size = 360000;
    CHECK_CUDA_ERRORS(cudaMemcpy(gRNPN, d_gRNPN, size * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynRNPN, d_inSynRNPN, 600 * sizeof(float), cudaMemcpyDeviceToHost));
    }

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice()
 {
    pushANStateToDevice();
    pushANSpikesToDevice();
    pushPNStateToDevice();
    pushPNSpikesToDevice();
    pushRNStateToDevice();
    pushRNSpikesToDevice();
    pushANANStateToDevice();
    pushPNANStateToDevice();
    pushPNPNStateToDevice();
    pushRNPNStateToDevice();
    }

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushANSpikesToDevice();
    pushPNSpikesToDevice();
    pushRNSpikesToDevice();
    }
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushANCurrentSpikesToDevice();
    pushPNCurrentSpikesToDevice();
    pushRNCurrentSpikesToDevice();
    }
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushANSpikeEventsToDevice();
    pushPNSpikeEventsToDevice();
    pushRNSpikeEventsToDevice();
    }
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushANCurrentSpikeEventsToDevice();
    pushPNCurrentSpikeEventsToDevice();
    pushRNCurrentSpikeEventsToDevice();
    }
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullANStateFromDevice();
    pullANSpikesFromDevice();
    pullPNStateFromDevice();
    pullPNSpikesFromDevice();
    pullRNStateFromDevice();
    pullRNSpikesFromDevice();
    pullANANStateFromDevice();
    pullPNANStateFromDevice();
    pullPNPNStateFromDevice();
    pullRNPNStateFromDevice();
    }

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
pullANSpikesFromDevice();
    pullPNSpikesFromDevice();
    pullRNSpikesFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
pullANCurrentSpikesFromDevice();
    pullPNCurrentSpikesFromDevice();
    pullRNCurrentSpikesFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntAN, d_glbSpkCntAN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntPN, d_glbSpkCntPN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntRN, d_glbSpkCntRN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    
// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
pullANSpikeEventsFromDevice();
    pullPNSpikeEventsFromDevice();
    pullRNSpikeEventsFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
pullANCurrentSpikeEventsFromDevice();
    pullPNCurrentSpikeEventsFromDevice();
    pullRNCurrentSpikeEventsFromDevice();
    }

    
// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void copySpikeEventNFromDevice()
 {
    
CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntAN, d_glbSpkCntEvntAN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntPN, d_glbSpkCntEvntPN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntEvntRN, d_glbSpkCntEvntRN, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    
// ------------------------------------------------------------------------
// the time stepping procedure (using GPU)
void stepTimeGPU()
 {
    
//model.padSumSynapseTrgN[model.synapseGrpN - 1] is 1664
    dim3 sThreads(64, 1);
    dim3 sGrid(26, 1);
    
    dim3 nThreads(64, 1);
    dim3 nGrid(23, 1);
    
    calcSynapses <<< sGrid, sThreads >>> (t);
    calcNeurons <<< nGrid, nThreads >>> (offsetRN, ratesRN, t);
    iT++;
    t= iT*DT;
    }

    