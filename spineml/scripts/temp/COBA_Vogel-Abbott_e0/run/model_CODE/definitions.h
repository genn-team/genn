

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

extern std::mt19937 rng;
extern std::uniform_real_distribution<float> standardUniformDistribution;
extern std::normal_distribution<float> standardNormalDistribution;
extern std::exponential_distribution<float> standardExponentialDistribution;

// ------------------------------------------------------------------------
// neuron variables

extern unsigned int * glbSpkCntExcitatory;
extern unsigned int * d_glbSpkCntExcitatory;
extern unsigned int * glbSpkExcitatory;
extern unsigned int * d_glbSpkExcitatory;
extern unsigned int spkQuePtrExcitatory;
extern scalar * t_spikeExcitatory;
extern scalar * d_t_spikeExcitatory;
extern scalar * vExcitatory;
extern scalar * d_vExcitatory;
extern unsigned int * _regimeIDExcitatory;
extern unsigned int * d__regimeIDExcitatory;
extern unsigned int * glbSpkCntInhibitory;
extern unsigned int * d_glbSpkCntInhibitory;
extern unsigned int * glbSpkInhibitory;
extern unsigned int * d_glbSpkInhibitory;
extern unsigned int spkQuePtrInhibitory;
extern scalar * t_spikeInhibitory;
extern scalar * d_t_spikeInhibitory;
extern scalar * vInhibitory;
extern scalar * d_vInhibitory;
extern unsigned int * _regimeIDInhibitory;
extern unsigned int * d__regimeIDInhibitory;
extern unsigned int * glbSpkCntSpike_Source;
extern unsigned int * d_glbSpkCntSpike_Source;
extern unsigned int * glbSpkSpike_Source;
extern unsigned int * d_glbSpkSpike_Source;

#define glbSpkShiftExcitatory spkQuePtrExcitatory*3200
#define glbSpkShiftInhibitory spkQuePtrInhibitory*800
#define glbSpkShiftSpike_Source 0
#define spikeCount_Excitatory glbSpkCntExcitatory[spkQuePtrExcitatory]
#define spike_Excitatory (glbSpkExcitatory+(spkQuePtrExcitatory*3200))
#define spikeCount_Inhibitory glbSpkCntInhibitory[spkQuePtrInhibitory]
#define spike_Inhibitory (glbSpkInhibitory+(spkQuePtrInhibitory*800))
#define spikeCount_Spike_Source glbSpkCntSpike_Source[0]
#define spike_Spike_Source glbSpkSpike_Source

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynExcitatory_to_Excitatory_Synapse_0_weight_update;
extern float * d_inSynExcitatory_to_Excitatory_Synapse_0_weight_update;
extern SparseProjection CExcitatory_to_Excitatory_Synapse_0_weight_update;
extern float * inSynExcitatory_to_Inhibitory_Synapse_0_weight_update;
extern float * d_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update;
extern SparseProjection CExcitatory_to_Inhibitory_Synapse_0_weight_update;
extern float * inSynInhibitory_to_Excitatory_Synapse_0_weight_update;
extern float * d_inSynInhibitory_to_Excitatory_Synapse_0_weight_update;
extern SparseProjection CInhibitory_to_Excitatory_Synapse_0_weight_update;
extern float * inSynInhibitory_to_Inhibitory_Synapse_0_weight_update;
extern float * d_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update;
extern SparseProjection CInhibitory_to_Inhibitory_Synapse_0_weight_update;
extern float * inSynSpike_Source_to_Excitatory_Synapse_0_weight_update;
extern float * d_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update;
extern SparseProjection CSpike_Source_to_Excitatory_Synapse_0_weight_update;
extern float * inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update;
extern float * d_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update;
extern SparseProjection CSpike_Source_to_Inhibitory_Synapse_0_weight_update;

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/

// ------------------------------------------------------------------------
// copying things to device

void pushExcitatoryStateToDevice(bool hostInitialisedOnly = false);
void pushExcitatorySpikesToDevice(bool hostInitialisedOnly = false);
void pushExcitatorySpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushExcitatoryCurrentSpikesToDevice();
void pushExcitatoryCurrentSpikeEventsToDevice();
void pushInhibitoryStateToDevice(bool hostInitialisedOnly = false);
void pushInhibitorySpikesToDevice(bool hostInitialisedOnly = false);
void pushInhibitorySpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushInhibitoryCurrentSpikesToDevice();
void pushInhibitoryCurrentSpikeEventsToDevice();
void pushSpike_SourceStateToDevice(bool hostInitialisedOnly = false);
void pushSpike_SourceSpikesToDevice(bool hostInitialisedOnly = false);
void pushSpike_SourceSpikeEventsToDevice(bool hostInitialisedOnly = false);
void pushSpike_SourceCurrentSpikesToDevice();
void pushSpike_SourceCurrentSpikeEventsToDevice();
#define pushExcitatory_to_Excitatory_Synapse_0_weight_updateToDevice pushExcitatory_to_Excitatory_Synapse_0_weight_updateStateToDevice
void pushExcitatory_to_Excitatory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushExcitatory_to_Inhibitory_Synapse_0_weight_updateToDevice pushExcitatory_to_Inhibitory_Synapse_0_weight_updateStateToDevice
void pushExcitatory_to_Inhibitory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushInhibitory_to_Excitatory_Synapse_0_weight_updateToDevice pushInhibitory_to_Excitatory_Synapse_0_weight_updateStateToDevice
void pushInhibitory_to_Excitatory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushInhibitory_to_Inhibitory_Synapse_0_weight_updateToDevice pushInhibitory_to_Inhibitory_Synapse_0_weight_updateStateToDevice
void pushInhibitory_to_Inhibitory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushSpike_Source_to_Excitatory_Synapse_0_weight_updateToDevice pushSpike_Source_to_Excitatory_Synapse_0_weight_updateStateToDevice
void pushSpike_Source_to_Excitatory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);
#define pushSpike_Source_to_Inhibitory_Synapse_0_weight_updateToDevice pushSpike_Source_to_Inhibitory_Synapse_0_weight_updateStateToDevice
void pushSpike_Source_to_Inhibitory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly = false);

// ------------------------------------------------------------------------
// copying things from device

void pullExcitatoryStateFromDevice();
void pullExcitatorySpikesFromDevice();
void pullExcitatorySpikeEventsFromDevice();
void pullExcitatoryCurrentSpikesFromDevice();
void pullExcitatoryCurrentSpikeEventsFromDevice();
void pullInhibitoryStateFromDevice();
void pullInhibitorySpikesFromDevice();
void pullInhibitorySpikeEventsFromDevice();
void pullInhibitoryCurrentSpikesFromDevice();
void pullInhibitoryCurrentSpikeEventsFromDevice();
void pullSpike_SourceStateFromDevice();
void pullSpike_SourceSpikesFromDevice();
void pullSpike_SourceSpikeEventsFromDevice();
void pullSpike_SourceCurrentSpikesFromDevice();
void pullSpike_SourceCurrentSpikeEventsFromDevice();
#define pullExcitatory_to_Excitatory_Synapse_0_weight_updateFromDevice pullExcitatory_to_Excitatory_Synapse_0_weight_updateStateFromDevice
void pullExcitatory_to_Excitatory_Synapse_0_weight_updateStateFromDevice();
#define pullExcitatory_to_Inhibitory_Synapse_0_weight_updateFromDevice pullExcitatory_to_Inhibitory_Synapse_0_weight_updateStateFromDevice
void pullExcitatory_to_Inhibitory_Synapse_0_weight_updateStateFromDevice();
#define pullInhibitory_to_Excitatory_Synapse_0_weight_updateFromDevice pullInhibitory_to_Excitatory_Synapse_0_weight_updateStateFromDevice
void pullInhibitory_to_Excitatory_Synapse_0_weight_updateStateFromDevice();
#define pullInhibitory_to_Inhibitory_Synapse_0_weight_updateFromDevice pullInhibitory_to_Inhibitory_Synapse_0_weight_updateStateFromDevice
void pullInhibitory_to_Inhibitory_Synapse_0_weight_updateStateFromDevice();
#define pullSpike_Source_to_Excitatory_Synapse_0_weight_updateFromDevice pullSpike_Source_to_Excitatory_Synapse_0_weight_updateStateFromDevice
void pullSpike_Source_to_Excitatory_Synapse_0_weight_updateStateFromDevice();
#define pullSpike_Source_to_Inhibitory_Synapse_0_weight_updateFromDevice pullSpike_Source_to_Inhibitory_Synapse_0_weight_updateStateFromDevice
void pullSpike_Source_to_Inhibitory_Synapse_0_weight_updateStateFromDevice();

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

void allocateExcitatory_to_Excitatory_Synapse_0_weight_update(unsigned int connN);

void allocateExcitatory_to_Inhibitory_Synapse_0_weight_update(unsigned int connN);

void allocateInhibitory_to_Excitatory_Synapse_0_weight_update(unsigned int connN);

void allocateInhibitory_to_Inhibitory_Synapse_0_weight_update(unsigned int connN);

void allocateSpike_Source_to_Excitatory_Synapse_0_weight_update(unsigned int connN);

void allocateSpike_Source_to_Inhibitory_Synapse_0_weight_update(unsigned int connN);

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
