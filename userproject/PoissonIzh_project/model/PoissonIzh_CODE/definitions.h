

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model PoissonIzh containing useful Macros used for both GPU amd CPU versions.
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
#define DT 1.000000f
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

extern std::mt19937 rng;
extern std::uniform_real_distribution<float> standardUniformDistribution;
extern std::normal_distribution<float> standardNormalDistribution;
extern std::exponential_distribution<float> standardExponentialDistribution;

// ------------------------------------------------------------------------
// neuron variables

extern unsigned int * glbSpkCntIzh1;
extern unsigned int * d_glbSpkCntIzh1;
extern unsigned int * glbSpkIzh1;
extern unsigned int * d_glbSpkIzh1;
extern scalar * VIzh1;
extern scalar * d_VIzh1;
extern scalar * UIzh1;
extern scalar * d_UIzh1;
extern unsigned int * glbSpkCntPN;
extern unsigned int * d_glbSpkCntPN;
extern unsigned int * glbSpkPN;
extern unsigned int * d_glbSpkPN;
extern curandState *d_rngPN;
extern scalar * timeStepToSpikePN;
extern scalar * d_timeStepToSpikePN;

#define glbSpkShiftIzh1 0
#define glbSpkShiftPN 0
#define spikeCount_Izh1 glbSpkCntIzh1[0]
#define spike_Izh1 glbSpkIzh1
#define spikeCount_PN glbSpkCntPN[0]
#define spike_PN glbSpkPN

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynPNIzh1;
extern float * d_inSynPNIzh1;
extern scalar * gPNIzh1;
extern scalar * d_gPNIzh1;

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/

// ------------------------------------------------------------------------
// copying things to device

void pushIzh1StateToDevice(bool hostInitialisedOnly = false);
void pushIzh1SpikesToDevice(bool hostInitialisedOnly = false);
void pushIzh1SpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushIzh1CurrentSpikesToDevice();
void pushIzh1CurrentSpikeEventsToDevice();
void pushPNStateToDevice(bool hostInitialisedOnly = false);
void pushPNSpikesToDevice(bool hostInitialisedOnly = false);
void pushPNSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushPNCurrentSpikesToDevice();
void pushPNCurrentSpikeEventsToDevice();
#define pushPNIzh1ToDevice pushPNIzh1StateToDevice
void pushPNIzh1StateToDevice(bool hostInitialisedOnly = false);

// ------------------------------------------------------------------------
// copying things from device

void pullIzh1StateFromDevice();
void pullIzh1SpikesFromDevice();
void pullIzh1SpikeEventsFromDevice();
void pullIzh1CurrentSpikesFromDevice();
void pullIzh1CurrentSpikeEventsFromDevice();
void pullPNStateFromDevice();
void pullPNSpikesFromDevice();
void pullPNSpikeEventsFromDevice();
void pullPNCurrentSpikesFromDevice();
void pullPNCurrentSpikeEventsFromDevice();
#define pullPNIzh1FromDevice pullPNIzh1StateFromDevice
void pullPNIzh1StateFromDevice();

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

void initPoissonIzh();

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
