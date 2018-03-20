

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model Schmuker_2014_classifier containing useful Macros used for both GPU amd CPU versions.
*/
//-------------------------------------------------------------------------

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "utils.h"
#include "sparseUtils.h"

#include "sparseProjection.h"
#include <stdint.h>

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
#define DT 0.500000f
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

extern unsigned int * glbSpkCntAN;
extern unsigned int * d_glbSpkCntAN;
extern unsigned int * glbSpkAN;
extern unsigned int * d_glbSpkAN;
extern unsigned int * glbSpkCntEvntAN;
extern unsigned int * d_glbSpkCntEvntAN;
extern unsigned int * glbSpkEvntAN;
extern unsigned int * d_glbSpkEvntAN;
extern scalar * VAN;
extern scalar * d_VAN;
extern scalar * preVAN;
extern scalar * d_preVAN;
extern unsigned int * glbSpkCntPN;
extern unsigned int * d_glbSpkCntPN;
extern unsigned int * glbSpkPN;
extern unsigned int * d_glbSpkPN;
extern unsigned int * glbSpkCntEvntPN;
extern unsigned int * d_glbSpkCntEvntPN;
extern unsigned int * glbSpkEvntPN;
extern unsigned int * d_glbSpkEvntPN;
extern scalar * VPN;
extern scalar * d_VPN;
extern scalar * preVPN;
extern scalar * d_preVPN;
extern unsigned int * glbSpkCntRN;
extern unsigned int * d_glbSpkCntRN;
extern unsigned int * glbSpkRN;
extern unsigned int * d_glbSpkRN;
extern unsigned int * glbSpkCntEvntRN;
extern unsigned int * d_glbSpkCntEvntRN;
extern unsigned int * glbSpkEvntRN;
extern unsigned int * d_glbSpkEvntRN;
extern scalar * VRN;
extern scalar * d_VRN;
extern uint64_t * seedRN;
extern uint64_t * d_seedRN;
extern scalar * spikeTimeRN;
extern scalar * d_spikeTimeRN;
extern uint64_t * ratesRN;
extern uint64_t * d_ratesRN;
extern unsigned int offsetRN;
extern unsigned int d_offsetRN;

#define glbSpkShiftAN 0
#define glbSpkShiftPN 0
#define glbSpkShiftRN 0
#define spikeCount_AN glbSpkCntAN[0]
#define spike_AN glbSpkAN
#define spikeEventCount_AN glbSpkCntEvntAN[0]
#define spikeEvent_AN glbSpkEvntAN
#define spikeCount_PN glbSpkCntPN[0]
#define spike_PN glbSpkPN
#define spikeEventCount_PN glbSpkCntEvntPN[0]
#define spikeEvent_PN glbSpkEvntPN
#define spikeCount_RN glbSpkCntRN[0]
#define spike_RN glbSpkRN
#define spikeEventCount_RN glbSpkCntEvntRN[0]
#define spikeEvent_RN glbSpkEvntRN

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynANAN;
extern float * d_inSynANAN;
extern scalar * gANAN;
extern scalar * d_gANAN;
extern float * inSynPNAN;
extern float * d_inSynPNAN;
extern scalar * gPNAN;
extern scalar * d_gPNAN;
extern float * inSynPNPN;
extern float * d_inSynPNPN;
extern scalar * gPNPN;
extern scalar * d_gPNPN;
extern float * inSynRNPN;
extern float * d_inSynRNPN;
extern scalar * gRNPN;
extern scalar * d_gRNPN;

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/

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
// copying things to device

void pushANStateToDevice();
void pushANSpikesToDevice();
void pushANSpikeEventsToDevice();
void pushANCurrentSpikesToDevice();
void pushANCurrentSpikeEventsToDevice();
void pushPNStateToDevice();
void pushPNSpikesToDevice();
void pushPNSpikeEventsToDevice();
void pushPNCurrentSpikesToDevice();
void pushPNCurrentSpikeEventsToDevice();
void pushRNStateToDevice();
void pushRNSpikesToDevice();
void pushRNSpikeEventsToDevice();
void pushRNCurrentSpikesToDevice();
void pushRNCurrentSpikeEventsToDevice();
#define pushANANToDevice pushANANStateToDevice
void pushANANStateToDevice();
#define pushPNANToDevice pushPNANStateToDevice
void pushPNANStateToDevice();
#define pushPNPNToDevice pushPNPNStateToDevice
void pushPNPNStateToDevice();
#define pushRNPNToDevice pushRNPNStateToDevice
void pushRNPNStateToDevice();

// ------------------------------------------------------------------------
// copying things from device

void pullANStateFromDevice();
void pullANSpikesFromDevice();
void pullANSpikeEventsFromDevice();
void pullANCurrentSpikesFromDevice();
void pullANCurrentSpikeEventsFromDevice();
void pullPNStateFromDevice();
void pullPNSpikesFromDevice();
void pullPNSpikeEventsFromDevice();
void pullPNCurrentSpikesFromDevice();
void pullPNCurrentSpikeEventsFromDevice();
void pullRNStateFromDevice();
void pullRNSpikesFromDevice();
void pullRNSpikeEventsFromDevice();
void pullRNCurrentSpikesFromDevice();
void pullRNCurrentSpikeEventsFromDevice();
#define pullANANFromDevice pullANANStateFromDevice
void pullANANStateFromDevice();
#define pullPNANFromDevice pullPNANStateFromDevice
void pullPNANStateFromDevice();
#define pullPNPNFromDevice pullPNPNStateFromDevice
void pullPNPNStateFromDevice();
#define pullRNPNFromDevice pullRNPNStateFromDevice
void pullRNPNStateFromDevice();

// ------------------------------------------------------------------------
// global copying values to device

void copyStateToDevice();

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

void initSchmuker_2014_classifier();

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
// Throw an error for "old style" time stepping calls (using CPU)

template <class T>
void stepTimeCPU(T arg1, ...) {
    gennError("Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.");
    }

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)

void stepTimeCPU();

// ------------------------------------------------------------------------
// Throw an error for "old style" time stepping calls (using GPU)

template <class T>
void stepTimeGPU(T arg1, ...) {
    gennError("Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.");
    }

// ------------------------------------------------------------------------
// the actual time stepping procedure (using GPU)

void stepTimeGPU();

#endif
