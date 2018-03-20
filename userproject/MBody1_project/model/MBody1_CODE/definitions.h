

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model MBody1 containing useful Macros used for both GPU amd CPU versions.
*/
//-------------------------------------------------------------------------

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "utils.h"
#include "sparseUtils.h"

#include "sparseProjection.h"
#include <cstdint>
#include <random>
#include <curand_kernel.h>

#ifndef CHECK_CUDA_ERRORS
#define CHECK_CUDA_ERRORS(call) {\
    cudaError_t error = call;\
    if (error != cudaSuccess) {\
        fprintf(stderr, "%s: %i: cuda error %i: %s\n", __FILE__, __LINE__, (int) error, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}
#endif

#undef DT
#define DT 0.100000f
#ifndef MYRAND
#define MYRAND(Y,X) Y = Y * 1103515245 + 12345; X = (Y >> 16);
#endif
#ifndef MYRAND_MAX
#define MYRAND_MAX 0x0000FFFFFFFFFFFFLL
#endif

#ifndef scalar
typedef float scalar;
#endif
#ifndef SCALAR_MIN
#define SCALAR_MIN 0.000000f
#endif
#ifndef SCALAR_MAX
#define SCALAR_MAX 340282346638528859811704183484516925440.000000f
#endif

// ------------------------------------------------------------------------
// global variables

extern unsigned long long iT;
extern float t;


// ------------------------------------------------------------------------
// neuron variables

extern unsigned int * glbSpkCntDN;
extern unsigned int * d_glbSpkCntDN;
extern unsigned int * glbSpkDN;
extern unsigned int * d_glbSpkDN;
extern unsigned int * glbSpkCntEvntDN;
extern unsigned int * d_glbSpkCntEvntDN;
extern unsigned int * glbSpkEvntDN;
extern unsigned int * d_glbSpkEvntDN;
extern float * sTDN;
extern float * d_sTDN;
extern scalar * VDN;
extern scalar * d_VDN;
extern scalar * mDN;
extern scalar * d_mDN;
extern scalar * hDN;
extern scalar * d_hDN;
extern scalar * nDN;
extern scalar * d_nDN;
extern unsigned int * glbSpkCntKC;
extern unsigned int * d_glbSpkCntKC;
extern unsigned int * glbSpkKC;
extern unsigned int * d_glbSpkKC;
extern float * sTKC;
extern float * d_sTKC;
extern scalar * VKC;
extern scalar * d_VKC;
extern scalar * mKC;
extern scalar * d_mKC;
extern scalar * hKC;
extern scalar * d_hKC;
extern scalar * nKC;
extern scalar * d_nKC;
extern unsigned int * glbSpkCntLHI;
extern unsigned int * d_glbSpkCntLHI;
extern unsigned int * glbSpkLHI;
extern unsigned int * d_glbSpkLHI;
extern unsigned int * glbSpkCntEvntLHI;
extern unsigned int * d_glbSpkCntEvntLHI;
extern unsigned int * glbSpkEvntLHI;
extern unsigned int * d_glbSpkEvntLHI;
extern scalar * VLHI;
extern scalar * d_VLHI;
extern scalar * mLHI;
extern scalar * d_mLHI;
extern scalar * hLHI;
extern scalar * d_hLHI;
extern scalar * nLHI;
extern scalar * d_nLHI;
extern unsigned int * glbSpkCntPN;
extern unsigned int * d_glbSpkCntPN;
extern unsigned int * glbSpkPN;
extern unsigned int * d_glbSpkPN;
extern scalar * VPN;
extern scalar * d_VPN;
extern uint64_t * seedPN;
extern uint64_t * d_seedPN;
extern scalar * spikeTimePN;
extern scalar * d_spikeTimePN;
extern uint64_t * ratesPN;
extern unsigned int offsetPN;

#define glbSpkShiftDN 0
#define glbSpkShiftKC 0
#define glbSpkShiftLHI 0
#define glbSpkShiftPN 0
#define spikeCount_DN glbSpkCntDN[0]
#define spike_DN glbSpkDN
#define spikeEventCount_DN glbSpkCntEvntDN[0]
#define spikeEvent_DN glbSpkEvntDN
#define spikeCount_KC glbSpkCntKC[0]
#define spike_KC glbSpkKC
#define spikeCount_LHI glbSpkCntLHI[0]
#define spike_LHI glbSpkLHI
#define spikeEventCount_LHI glbSpkCntEvntLHI[0]
#define spikeEvent_LHI glbSpkEvntLHI
#define spikeCount_PN glbSpkCntPN[0]
#define spike_PN glbSpkPN

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynDNDN;
extern float * d_inSynDNDN;
extern float * inSynKCDN;
extern float * d_inSynKCDN;
extern scalar * gKCDN;
extern scalar * d_gKCDN;
extern scalar * gRawKCDN;
extern scalar * d_gRawKCDN;
extern float * inSynLHIKC;
extern float * d_inSynLHIKC;
extern float * inSynPNKC;
extern float * d_inSynPNKC;
extern scalar * gPNKC;
extern scalar * d_gPNKC;
extern float * inSynPNLHI;
extern float * d_inSynPNLHI;
extern scalar * gPNLHI;
extern scalar * d_gPNLHI;

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/

// ------------------------------------------------------------------------
// copying things to device

void pushDNStateToDevice(bool hostInitialisedOnly = false);
void pushDNSpikesToDevice(bool hostInitialisedOnly = false);
void pushDNSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushDNCurrentSpikesToDevice();
void pushDNCurrentSpikeEventsToDevice();
void pushKCStateToDevice(bool hostInitialisedOnly = false);
void pushKCSpikesToDevice(bool hostInitialisedOnly = false);
void pushKCSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushKCCurrentSpikesToDevice();
void pushKCCurrentSpikeEventsToDevice();
void pushLHIStateToDevice(bool hostInitialisedOnly = false);
void pushLHISpikesToDevice(bool hostInitialisedOnly = false);
void pushLHISpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushLHICurrentSpikesToDevice();
void pushLHICurrentSpikeEventsToDevice();
void pushPNStateToDevice(bool hostInitialisedOnly = false);
void pushPNSpikesToDevice(bool hostInitialisedOnly = false);
void pushPNSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushPNCurrentSpikesToDevice();
void pushPNCurrentSpikeEventsToDevice();
#define pushDNDNToDevice pushDNDNStateToDevice
void pushDNDNStateToDevice(bool hostInitialisedOnly = false);
#define pushKCDNToDevice pushKCDNStateToDevice
void pushKCDNStateToDevice(bool hostInitialisedOnly = false);
#define pushLHIKCToDevice pushLHIKCStateToDevice
void pushLHIKCStateToDevice(bool hostInitialisedOnly = false);
#define pushPNKCToDevice pushPNKCStateToDevice
void pushPNKCStateToDevice(bool hostInitialisedOnly = false);
#define pushPNLHIToDevice pushPNLHIStateToDevice
void pushPNLHIStateToDevice(bool hostInitialisedOnly = false);

// ------------------------------------------------------------------------
// copying things from device

void pullDNStateFromDevice();
void pullDNSpikesFromDevice();
void pullDNSpikeEventsFromDevice();
void pullDNCurrentSpikesFromDevice();
void pullDNCurrentSpikeEventsFromDevice();
void pullKCStateFromDevice();
void pullKCSpikesFromDevice();
void pullKCSpikeEventsFromDevice();
void pullKCCurrentSpikesFromDevice();
void pullKCCurrentSpikeEventsFromDevice();
void pullLHIStateFromDevice();
void pullLHISpikesFromDevice();
void pullLHISpikeEventsFromDevice();
void pullLHICurrentSpikesFromDevice();
void pullLHICurrentSpikeEventsFromDevice();
void pullPNStateFromDevice();
void pullPNSpikesFromDevice();
void pullPNSpikeEventsFromDevice();
void pullPNCurrentSpikesFromDevice();
void pullPNCurrentSpikeEventsFromDevice();
#define pullDNDNFromDevice pullDNDNStateFromDevice
void pullDNDNStateFromDevice();
#define pullKCDNFromDevice pullKCDNStateFromDevice
void pullKCDNStateFromDevice();
#define pullLHIKCFromDevice pullLHIKCStateFromDevice
void pullLHIKCStateFromDevice();
#define pullPNKCFromDevice pullPNKCStateFromDevice
void pullPNKCStateFromDevice();
#define pullPNLHIFromDevice pullPNLHIStateFromDevice
void pullPNLHIStateFromDevice();

// ------------------------------------------------------------------------
// global copying values to device

void copyStateToDevice(bool hostInitialisedOnly = false);

// ------------------------------------------------------------------------
// global copying spikes to device

void copySpikesToDevice();

// ------------------------------------------------------------------------
// copying current spikes to device

void copyCurrentSpikesToDevice();

// ------------------------------------------------------------------------
// global copying spike events to device

void copySpikeEventsToDevice();

// ------------------------------------------------------------------------
// copying current spikes to device

void copyCurrentSpikeEventsToDevice();

// ------------------------------------------------------------------------
// global copying values from device

void copyStateFromDevice();

// ------------------------------------------------------------------------
// global copying spikes from device

void copySpikesFromDevice();

// ------------------------------------------------------------------------
// copying current spikes from device

void copyCurrentSpikesFromDevice();

// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)

void copySpikeNFromDevice();

// ------------------------------------------------------------------------
// global copying spikeEvents from device

void copySpikeEventsFromDevice();

// ------------------------------------------------------------------------
// copying current spikeEvents from device

void copyCurrentSpikeEventsFromDevice();

// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)

void copySpikeEventNFromDevice();

// ------------------------------------------------------------------------
// Function for setting the CUDA device and the host's global variables.
// Also estimates memory usage on device.

void allocateMem();

// ------------------------------------------------------------------------
// Function to (re)set all model variables to their compile-time, homogeneous initial
// values. Note that this typically includes synaptic weight values. The function
// (re)sets host side variables and copies them to the GPU device.

void initialize();

void initializeAllSparseArrays();

// ------------------------------------------------------------------------
// initialization of variables, e.g. reverse sparse arrays etc.
// that the user would not want to worry about

void initMBody1();

// ------------------------------------------------------------------------
// Function to free all global memory structures.

void freeMem();

//-------------------------------------------------------------------------
// Function to convert a firing probability (per time step) to an integer of type uint64_t
// that can be used as a threshold for the GeNN random number generator to generate events with the given probability.

void convertProbabilityToRandomNumberThreshold(float *p_pattern, uint64_t *pattern, int N);

//-------------------------------------------------------------------------
// Function to convert a firing rate (in kHz) to an integer of type uint64_t that can be used
// as a threshold for the GeNN random number generator to generate events with the given rate.

void convertRateToRandomNumberThreshold(float *rateKHz_pattern, uint64_t *pattern, int N);

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)

void stepTimeCPU();

// ------------------------------------------------------------------------
// the actual time stepping procedure (using GPU)

void stepTimeGPU();

// ------------------------------------------------------------------------
// Throw an error for "old style" time stepping calls (using GPU)

template <class T>
void stepTimeGPU(T arg1, ...) {
    gennError("Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.");
}

// ------------------------------------------------------------------------
// Helper function for allocating memory blocks on the GPU device

template<class T>
void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)
{
    void *devptr;
    CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));
    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));
    CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------
// Helper function for getting the device pointer corresponding to a zero-copied host pointer and assigning it to a symbol

template<class T>
void deviceZeroCopy(T hostPtr, const T *devPtr, const T &devSymbol)
{
    CHECK_CUDA_ERRORS(cudaHostGetDevicePointer((void **)devPtr, (void*)hostPtr, 0));
    void *devSymbolPtr;
    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymbolPtr, devSymbol));
    CHECK_CUDA_ERRORS(cudaMemcpy(devSymbolPtr, devPtr, sizeof(void*), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------
// Throw an error for "old style" time stepping calls (using CPU)

template <class T>
void stepTimeCPU(T arg1, ...) {
    gennError("Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.");
}

#endif
