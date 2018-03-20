

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model Izh_sparse containing useful Macros used for both GPU amd CPU versions.
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
extern curandStatePhilox4_32_10_t *d_rng;

// ------------------------------------------------------------------------
// neuron variables

extern unsigned int * glbSpkCntPExc;
extern unsigned int * d_glbSpkCntPExc;
extern unsigned int * glbSpkPExc;
extern unsigned int * d_glbSpkPExc;
extern curandState *d_rngPExc;
extern scalar * VPExc;
extern scalar * d_VPExc;
extern scalar * UPExc;
extern scalar * d_UPExc;
extern scalar * aPExc;
extern scalar * d_aPExc;
extern scalar * bPExc;
extern scalar * d_bPExc;
extern scalar * cPExc;
extern scalar * d_cPExc;
extern scalar * dPExc;
extern scalar * d_dPExc;
extern unsigned int * glbSpkCntPInh;
extern unsigned int * d_glbSpkCntPInh;
extern unsigned int * glbSpkPInh;
extern unsigned int * d_glbSpkPInh;
extern curandState *d_rngPInh;
extern scalar * VPInh;
extern scalar * d_VPInh;
extern scalar * UPInh;
extern scalar * d_UPInh;
extern scalar * aPInh;
extern scalar * d_aPInh;
extern scalar * bPInh;
extern scalar * d_bPInh;
extern scalar * cPInh;
extern scalar * d_cPInh;
extern scalar * dPInh;
extern scalar * d_dPInh;

#define glbSpkShiftPExc 0
#define glbSpkShiftPInh 0
#define spikeCount_PExc glbSpkCntPExc[0]
#define spike_PExc glbSpkPExc
#define spikeCount_PInh glbSpkCntPInh[0]
#define spike_PInh glbSpkPInh

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynExc_Exc;
extern float * d_inSynExc_Exc;
extern SparseProjection CExc_Exc;
extern scalar * gExc_Exc;
extern scalar * d_gExc_Exc;
extern float * inSynExc_Inh;
extern float * d_inSynExc_Inh;
extern SparseProjection CExc_Inh;
extern scalar * gExc_Inh;
extern scalar * d_gExc_Inh;
extern float * inSynInh_Exc;
extern float * d_inSynInh_Exc;
extern SparseProjection CInh_Exc;
extern scalar * gInh_Exc;
extern scalar * d_gInh_Exc;
extern float * inSynInh_Inh;
extern float * d_inSynInh_Inh;
extern SparseProjection CInh_Inh;
extern scalar * gInh_Inh;
extern scalar * d_gInh_Inh;

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/

// ------------------------------------------------------------------------
// copying things to device

void pushPExcStateToDevice(bool hostInitialisedOnly = false);
void pushPExcSpikesToDevice(bool hostInitialisedOnly = false);
void pushPExcSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushPExcCurrentSpikesToDevice();
void pushPExcCurrentSpikeEventsToDevice();
void pushPInhStateToDevice(bool hostInitialisedOnly = false);
void pushPInhSpikesToDevice(bool hostInitialisedOnly = false);
void pushPInhSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushPInhCurrentSpikesToDevice();
void pushPInhCurrentSpikeEventsToDevice();
#define pushExc_ExcToDevice pushExc_ExcStateToDevice
void pushExc_ExcStateToDevice(bool hostInitialisedOnly = false);
#define pushExc_InhToDevice pushExc_InhStateToDevice
void pushExc_InhStateToDevice(bool hostInitialisedOnly = false);
#define pushInh_ExcToDevice pushInh_ExcStateToDevice
void pushInh_ExcStateToDevice(bool hostInitialisedOnly = false);
#define pushInh_InhToDevice pushInh_InhStateToDevice
void pushInh_InhStateToDevice(bool hostInitialisedOnly = false);

// ------------------------------------------------------------------------
// copying things from device

void pullPExcStateFromDevice();
void pullPExcSpikesFromDevice();
void pullPExcSpikeEventsFromDevice();
void pullPExcCurrentSpikesFromDevice();
void pullPExcCurrentSpikeEventsFromDevice();
void pullPInhStateFromDevice();
void pullPInhSpikesFromDevice();
void pullPInhSpikeEventsFromDevice();
void pullPInhCurrentSpikesFromDevice();
void pullPInhCurrentSpikeEventsFromDevice();
#define pullExc_ExcFromDevice pullExc_ExcStateFromDevice
void pullExc_ExcStateFromDevice();
#define pullExc_InhFromDevice pullExc_InhStateFromDevice
void pullExc_InhStateFromDevice();
#define pullInh_ExcFromDevice pullInh_ExcStateFromDevice
void pullInh_ExcStateFromDevice();
#define pullInh_InhFromDevice pullInh_InhStateFromDevice
void pullInh_InhStateFromDevice();

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

void allocateExc_Exc(unsigned int connN);

void allocateExc_Inh(unsigned int connN);

void allocateInh_Exc(unsigned int connN);

void allocateInh_Inh(unsigned int connN);

// ------------------------------------------------------------------------
// Function to (re)set all model variables to their compile-time, homogeneous initial
// values. Note that this typically includes synaptic weight values. The function
// (re)sets host side variables and copies them to the GPU device.

void initialize();

void initializeAllSparseArrays();

// ------------------------------------------------------------------------
// initialization of variables, e.g. reverse sparse arrays etc.
// that the user would not want to worry about

void initIzh_sparse();

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
