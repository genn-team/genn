

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model model containing useful Macros used for both GPU amd CPU versions.
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

extern "C" {
// ------------------------------------------------------------------------
// global variables

extern unsigned long long iT;
extern float t;


// ------------------------------------------------------------------------
// neuron variables

extern unsigned int * glbSpkCntCortex;
extern unsigned int * d_glbSpkCntCortex;
extern unsigned int * glbSpkCortex;
extern unsigned int * d_glbSpkCortex;
extern scalar * aCortex;
extern scalar * d_aCortex;
extern scalar * inCortex;
extern scalar * d_inCortex;
extern scalar * outCortex;
extern scalar * d_outCortex;
extern unsigned int * glbSpkCntD1;
extern unsigned int * d_glbSpkCntD1;
extern unsigned int * glbSpkD1;
extern unsigned int * d_glbSpkD1;
extern scalar * aD1;
extern scalar * d_aD1;
extern scalar * outD1;
extern scalar * d_outD1;
extern unsigned int * glbSpkCntD2;
extern unsigned int * d_glbSpkCntD2;
extern unsigned int * glbSpkD2;
extern unsigned int * d_glbSpkD2;
extern scalar * aD2;
extern scalar * d_aD2;
extern scalar * outD2;
extern scalar * d_outD2;
extern unsigned int * glbSpkCntGPe;
extern unsigned int * d_glbSpkCntGPe;
extern unsigned int * glbSpkGPe;
extern unsigned int * d_glbSpkGPe;
extern unsigned int spkQuePtrGPe;
extern scalar * aGPe;
extern scalar * d_aGPe;
extern scalar * outGPe;
extern scalar * d_outGPe;
extern unsigned int * glbSpkCntGPi;
extern unsigned int * d_glbSpkCntGPi;
extern unsigned int * glbSpkGPi;
extern unsigned int * d_glbSpkGPi;
extern scalar * aGPi;
extern scalar * d_aGPi;
extern scalar * outGPi;
extern scalar * d_outGPi;
extern unsigned int * glbSpkCntSTN;
extern unsigned int * d_glbSpkCntSTN;
extern unsigned int * glbSpkSTN;
extern unsigned int * d_glbSpkSTN;
extern scalar * aSTN;
extern scalar * d_aSTN;
extern scalar * outSTN;
extern scalar * d_outSTN;

#define glbSpkShiftCortex 0
#define glbSpkShiftD1 0
#define glbSpkShiftD2 0
#define glbSpkShiftGPe spkQuePtrGPe*6
#define glbSpkShiftGPi 0
#define glbSpkShiftSTN 0
#define spikeCount_Cortex glbSpkCntCortex[0]
#define spike_Cortex glbSpkCortex
#define spikeCount_D1 glbSpkCntD1[0]
#define spike_D1 glbSpkD1
#define spikeCount_D2 glbSpkCntD2[0]
#define spike_D2 glbSpkD2
#define spikeCount_GPe glbSpkCntGPe[0]
#define spike_GPe glbSpkGPe
#define spikeCount_GPi glbSpkCntGPi[0]
#define spike_GPi glbSpkGPi
#define spikeCount_STN glbSpkCntSTN[0]
#define spike_STN glbSpkSTN

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynCortex_to_D1_Synapse_0_weight_update;
extern float * d_inSynCortex_to_D1_Synapse_0_weight_update;
extern SparseProjection CCortex_to_D1_Synapse_0_weight_update;
extern float * inSynCortex_to_D2_Synapse_0_weight_update;
extern float * d_inSynCortex_to_D2_Synapse_0_weight_update;
extern SparseProjection CCortex_to_D2_Synapse_0_weight_update;
extern float * inSynCortex_to_STN_Synapse_0_weight_update;
extern float * d_inSynCortex_to_STN_Synapse_0_weight_update;
extern SparseProjection CCortex_to_STN_Synapse_0_weight_update;
extern float * inSynD1_to_GPi_Synapse_0_weight_update;
extern float * d_inSynD1_to_GPi_Synapse_0_weight_update;
extern SparseProjection CD1_to_GPi_Synapse_0_weight_update;
extern float * inSynD2_to_GPe_Synapse_0_weight_update;
extern float * d_inSynD2_to_GPe_Synapse_0_weight_update;
extern SparseProjection CD2_to_GPe_Synapse_0_weight_update;
extern float * inSynGPe_to_GPi_Synapse_0_weight_update;
extern float * d_inSynGPe_to_GPi_Synapse_0_weight_update;
extern SparseProjection CGPe_to_GPi_Synapse_0_weight_update;
extern float * inSynGPe_to_STN_Synapse_0_weight_update;
extern float * d_inSynGPe_to_STN_Synapse_0_weight_update;
extern SparseProjection CGPe_to_STN_Synapse_0_weight_update;
extern float * inSynSTN_to_GPe_Synapse_0_weight_update;
extern float * d_inSynSTN_to_GPe_Synapse_0_weight_update;
extern float * inSynSTN_to_GPi_Synapse_0_weight_update;
extern float * d_inSynSTN_to_GPi_Synapse_0_weight_update;

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/

// ------------------------------------------------------------------------
// copying things to device

void pushCortexStateToDevice(bool hostInitialisedOnly = false);
void pushCortexSpikesToDevice(bool hostInitialisedOnly = false);
void pushCortexSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushCortexCurrentSpikesToDevice();
void pushCortexCurrentSpikeEventsToDevice();
void pushD1StateToDevice(bool hostInitialisedOnly = false);
void pushD1SpikesToDevice(bool hostInitialisedOnly = false);
void pushD1SpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushD1CurrentSpikesToDevice();
void pushD1CurrentSpikeEventsToDevice();
void pushD2StateToDevice(bool hostInitialisedOnly = false);
void pushD2SpikesToDevice(bool hostInitialisedOnly = false);
void pushD2SpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushD2CurrentSpikesToDevice();
void pushD2CurrentSpikeEventsToDevice();
void pushGPeStateToDevice(bool hostInitialisedOnly = false);
void pushGPeSpikesToDevice(bool hostInitialisedOnly = false);
void pushGPeSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushGPeCurrentSpikesToDevice();
void pushGPeCurrentSpikeEventsToDevice();
void pushGPiStateToDevice(bool hostInitialisedOnly = false);
void pushGPiSpikesToDevice(bool hostInitialisedOnly = false);
void pushGPiSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushGPiCurrentSpikesToDevice();
void pushGPiCurrentSpikeEventsToDevice();
void pushSTNStateToDevice(bool hostInitialisedOnly = false);
void pushSTNSpikesToDevice(bool hostInitialisedOnly = false);
void pushSTNSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushSTNCurrentSpikesToDevice();
void pushSTNCurrentSpikeEventsToDevice();
#define pushCortex_to_D1_Synapse_0_weight_updateToDevice pushCortex_to_D1_Synapse_0_weight_updateStateToDevice
void pushCortex_to_D1_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushCortex_to_D2_Synapse_0_weight_updateToDevice pushCortex_to_D2_Synapse_0_weight_updateStateToDevice
void pushCortex_to_D2_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushCortex_to_STN_Synapse_0_weight_updateToDevice pushCortex_to_STN_Synapse_0_weight_updateStateToDevice
void pushCortex_to_STN_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushD1_to_GPi_Synapse_0_weight_updateToDevice pushD1_to_GPi_Synapse_0_weight_updateStateToDevice
void pushD1_to_GPi_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushD2_to_GPe_Synapse_0_weight_updateToDevice pushD2_to_GPe_Synapse_0_weight_updateStateToDevice
void pushD2_to_GPe_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushGPe_to_GPi_Synapse_0_weight_updateToDevice pushGPe_to_GPi_Synapse_0_weight_updateStateToDevice
void pushGPe_to_GPi_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushGPe_to_STN_Synapse_0_weight_updateToDevice pushGPe_to_STN_Synapse_0_weight_updateStateToDevice
void pushGPe_to_STN_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushSTN_to_GPe_Synapse_0_weight_updateToDevice pushSTN_to_GPe_Synapse_0_weight_updateStateToDevice
void pushSTN_to_GPe_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushSTN_to_GPi_Synapse_0_weight_updateToDevice pushSTN_to_GPi_Synapse_0_weight_updateStateToDevice
void pushSTN_to_GPi_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);

// ------------------------------------------------------------------------
// copying things from device

void pullCortexStateFromDevice();
void pullCortexSpikesFromDevice();
void pullCortexSpikeEventsFromDevice();
void pullCortexCurrentSpikesFromDevice();
void pullCortexCurrentSpikeEventsFromDevice();
void pullD1StateFromDevice();
void pullD1SpikesFromDevice();
void pullD1SpikeEventsFromDevice();
void pullD1CurrentSpikesFromDevice();
void pullD1CurrentSpikeEventsFromDevice();
void pullD2StateFromDevice();
void pullD2SpikesFromDevice();
void pullD2SpikeEventsFromDevice();
void pullD2CurrentSpikesFromDevice();
void pullD2CurrentSpikeEventsFromDevice();
void pullGPeStateFromDevice();
void pullGPeSpikesFromDevice();
void pullGPeSpikeEventsFromDevice();
void pullGPeCurrentSpikesFromDevice();
void pullGPeCurrentSpikeEventsFromDevice();
void pullGPiStateFromDevice();
void pullGPiSpikesFromDevice();
void pullGPiSpikeEventsFromDevice();
void pullGPiCurrentSpikesFromDevice();
void pullGPiCurrentSpikeEventsFromDevice();
void pullSTNStateFromDevice();
void pullSTNSpikesFromDevice();
void pullSTNSpikeEventsFromDevice();
void pullSTNCurrentSpikesFromDevice();
void pullSTNCurrentSpikeEventsFromDevice();
#define pullCortex_to_D1_Synapse_0_weight_updateFromDevice pullCortex_to_D1_Synapse_0_weight_updateStateFromDevice
void pullCortex_to_D1_Synapse_0_weight_updateStateFromDevice();
#define pullCortex_to_D2_Synapse_0_weight_updateFromDevice pullCortex_to_D2_Synapse_0_weight_updateStateFromDevice
void pullCortex_to_D2_Synapse_0_weight_updateStateFromDevice();
#define pullCortex_to_STN_Synapse_0_weight_updateFromDevice pullCortex_to_STN_Synapse_0_weight_updateStateFromDevice
void pullCortex_to_STN_Synapse_0_weight_updateStateFromDevice();
#define pullD1_to_GPi_Synapse_0_weight_updateFromDevice pullD1_to_GPi_Synapse_0_weight_updateStateFromDevice
void pullD1_to_GPi_Synapse_0_weight_updateStateFromDevice();
#define pullD2_to_GPe_Synapse_0_weight_updateFromDevice pullD2_to_GPe_Synapse_0_weight_updateStateFromDevice
void pullD2_to_GPe_Synapse_0_weight_updateStateFromDevice();
#define pullGPe_to_GPi_Synapse_0_weight_updateFromDevice pullGPe_to_GPi_Synapse_0_weight_updateStateFromDevice
void pullGPe_to_GPi_Synapse_0_weight_updateStateFromDevice();
#define pullGPe_to_STN_Synapse_0_weight_updateFromDevice pullGPe_to_STN_Synapse_0_weight_updateStateFromDevice
void pullGPe_to_STN_Synapse_0_weight_updateStateFromDevice();
#define pullSTN_to_GPe_Synapse_0_weight_updateFromDevice pullSTN_to_GPe_Synapse_0_weight_updateStateFromDevice
void pullSTN_to_GPe_Synapse_0_weight_updateStateFromDevice();
#define pullSTN_to_GPi_Synapse_0_weight_updateFromDevice pullSTN_to_GPi_Synapse_0_weight_updateStateFromDevice
void pullSTN_to_GPi_Synapse_0_weight_updateStateFromDevice();

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

void allocateCortex_to_D1_Synapse_0_weight_update(unsigned int connN);

void allocateCortex_to_D2_Synapse_0_weight_update(unsigned int connN);

void allocateCortex_to_STN_Synapse_0_weight_update(unsigned int connN);

void allocateD1_to_GPi_Synapse_0_weight_update(unsigned int connN);

void allocateD2_to_GPe_Synapse_0_weight_update(unsigned int connN);

void allocateGPe_to_GPi_Synapse_0_weight_update(unsigned int connN);

void allocateGPe_to_STN_Synapse_0_weight_update(unsigned int connN);

// ------------------------------------------------------------------------
// Function to (re)set all model variables to their compile-time, homogeneous initial
// values. Note that this typically includes synaptic weight values. The function
// (re)sets host side variables and copies them to the GPU device.

void initialize();

void initializeAllSparseArrays();

// ------------------------------------------------------------------------
// initialization of variables, e.g. reverse sparse arrays etc.
// that the user would not want to worry about

void initmodel();

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

}	// extern "C"
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
