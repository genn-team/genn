

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model model containing general control code.
*/
//-------------------------------------------------------------------------

#define RUNNER_CC_COMPILE

#include "definitions.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cassert>
#include <stdint.h>
#if defined(__GNUG__) && !defined(__clang__) && defined(__x86_64__)
    #include <gnu/libc-version.h>
#endif

// ------------------------------------------------------------------------
// global variables

extern "C" {
unsigned long long iT;
float t;

std::mt19937 rng;
std::uniform_real_distribution<float> standardUniformDistribution(0.000000f, 1.000000f);
std::normal_distribution<float> standardNormalDistribution(0.000000f, 1.000000f);
std::exponential_distribution<float> standardExponentialDistribution(1.000000f);

// ------------------------------------------------------------------------
// neuron variables

__device__ volatile unsigned int d_done;
unsigned int * glbSpkCntExcitatory;
unsigned int * d_glbSpkCntExcitatory;
__device__ unsigned int * dd_glbSpkCntExcitatory;
unsigned int * glbSpkExcitatory;
unsigned int * d_glbSpkExcitatory;
__device__ unsigned int * dd_glbSpkExcitatory;
unsigned int spkQuePtrExcitatory;
__device__ volatile unsigned int dd_spkQuePtrExcitatory;
scalar * t_spikeExcitatory;
scalar * d_t_spikeExcitatory;
__device__ scalar * dd_t_spikeExcitatory;
scalar * vExcitatory;
scalar * d_vExcitatory;
__device__ scalar * dd_vExcitatory;
unsigned int * _regimeIDExcitatory;
unsigned int * d__regimeIDExcitatory;
__device__ unsigned int * dd__regimeIDExcitatory;
unsigned int * glbSpkCntInhibitory;
unsigned int * d_glbSpkCntInhibitory;
__device__ unsigned int * dd_glbSpkCntInhibitory;
unsigned int * glbSpkInhibitory;
unsigned int * d_glbSpkInhibitory;
__device__ unsigned int * dd_glbSpkInhibitory;
unsigned int spkQuePtrInhibitory;
__device__ volatile unsigned int dd_spkQuePtrInhibitory;
scalar * t_spikeInhibitory;
scalar * d_t_spikeInhibitory;
__device__ scalar * dd_t_spikeInhibitory;
scalar * vInhibitory;
scalar * d_vInhibitory;
__device__ scalar * dd_vInhibitory;
unsigned int * _regimeIDInhibitory;
unsigned int * d__regimeIDInhibitory;
__device__ unsigned int * dd__regimeIDInhibitory;
unsigned int * glbSpkCntSpike_Source;
unsigned int * d_glbSpkCntSpike_Source;
__device__ unsigned int * dd_glbSpkCntSpike_Source;
unsigned int * glbSpkSpike_Source;
unsigned int * d_glbSpkSpike_Source;
__device__ unsigned int * dd_glbSpkSpike_Source;

// ------------------------------------------------------------------------
// synapse variables

float * inSynExcitatory_to_Excitatory_Synapse_0_weight_update;
float * d_inSynExcitatory_to_Excitatory_Synapse_0_weight_update;
__device__ float * dd_inSynExcitatory_to_Excitatory_Synapse_0_weight_update;
SparseProjection CExcitatory_to_Excitatory_Synapse_0_weight_update;
unsigned int *d_indInGExcitatory_to_Excitatory_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGExcitatory_to_Excitatory_Synapse_0_weight_update;
unsigned int *d_indExcitatory_to_Excitatory_Synapse_0_weight_update;
__device__ unsigned int *dd_indExcitatory_to_Excitatory_Synapse_0_weight_update;
float * inSynExcitatory_to_Inhibitory_Synapse_0_weight_update;
float * d_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update;
__device__ float * dd_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update;
SparseProjection CExcitatory_to_Inhibitory_Synapse_0_weight_update;
unsigned int *d_indInGExcitatory_to_Inhibitory_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGExcitatory_to_Inhibitory_Synapse_0_weight_update;
unsigned int *d_indExcitatory_to_Inhibitory_Synapse_0_weight_update;
__device__ unsigned int *dd_indExcitatory_to_Inhibitory_Synapse_0_weight_update;
float * inSynInhibitory_to_Excitatory_Synapse_0_weight_update;
float * d_inSynInhibitory_to_Excitatory_Synapse_0_weight_update;
__device__ float * dd_inSynInhibitory_to_Excitatory_Synapse_0_weight_update;
SparseProjection CInhibitory_to_Excitatory_Synapse_0_weight_update;
unsigned int *d_indInGInhibitory_to_Excitatory_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGInhibitory_to_Excitatory_Synapse_0_weight_update;
unsigned int *d_indInhibitory_to_Excitatory_Synapse_0_weight_update;
__device__ unsigned int *dd_indInhibitory_to_Excitatory_Synapse_0_weight_update;
float * inSynInhibitory_to_Inhibitory_Synapse_0_weight_update;
float * d_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update;
__device__ float * dd_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update;
SparseProjection CInhibitory_to_Inhibitory_Synapse_0_weight_update;
unsigned int *d_indInGInhibitory_to_Inhibitory_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGInhibitory_to_Inhibitory_Synapse_0_weight_update;
unsigned int *d_indInhibitory_to_Inhibitory_Synapse_0_weight_update;
__device__ unsigned int *dd_indInhibitory_to_Inhibitory_Synapse_0_weight_update;
float * inSynSpike_Source_to_Excitatory_Synapse_0_weight_update;
float * d_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update;
__device__ float * dd_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update;
SparseProjection CSpike_Source_to_Excitatory_Synapse_0_weight_update;
unsigned int *d_indInGSpike_Source_to_Excitatory_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGSpike_Source_to_Excitatory_Synapse_0_weight_update;
unsigned int *d_indSpike_Source_to_Excitatory_Synapse_0_weight_update;
__device__ unsigned int *dd_indSpike_Source_to_Excitatory_Synapse_0_weight_update;
float * inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update;
float * d_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update;
__device__ float * dd_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update;
SparseProjection CSpike_Source_to_Inhibitory_Synapse_0_weight_update;
unsigned int *d_indInGSpike_Source_to_Inhibitory_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGSpike_Source_to_Inhibitory_Synapse_0_weight_update;
unsigned int *d_indSpike_Source_to_Inhibitory_Synapse_0_weight_update;
__device__ unsigned int *dd_indSpike_Source_to_Inhibitory_Synapse_0_weight_update;

}	// extern "C"
//-------------------------------------------------------------------------
/*! \brief Function to convert a firing probability (per time step) 
to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given probability.
*/
//-------------------------------------------------------------------------

void convertProbabilityToRandomNumberThreshold(float *p_pattern, uint64_t *pattern, int N)
{
    float fac= pow(2.0, (double) sizeof(uint64_t)*8-16);
    for (int i= 0; i < N; i++) {
        pattern[i]= (uint64_t) (p_pattern[i]*fac);
    }
}

//-------------------------------------------------------------------------
/*! \brief Function to convert a firing rate (in kHz) 
to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given rate.
*/
//-------------------------------------------------------------------------

void convertRateToRandomNumberThreshold(float *rateKHz_pattern, uint64_t *pattern, int N)
{
    float fac= pow(2.0, (double) sizeof(uint64_t)*8-16)*DT;
    for (int i= 0; i < N; i++) {
        pattern[i]= (uint64_t) (rateKHz_pattern[i]*fac);
    }
}

#include "runnerGPU.cc"

#include "init.cc"
#include "neuronFnct.cc"
#include "synapseFnct.cc"
void allocateMem()
{
    CHECK_CUDA_ERRORS(cudaSetDevice(0));
    cudaHostAlloc(&glbSpkCntExcitatory, 2 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntExcitatory, dd_glbSpkCntExcitatory, 2 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkExcitatory, 6400 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkExcitatory, dd_glbSpkExcitatory, 6400 * sizeof(unsigned int));
    cudaHostAlloc(&t_spikeExcitatory, 3200 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_t_spikeExcitatory, dd_t_spikeExcitatory, 3200 * sizeof(scalar));
    cudaHostAlloc(&vExcitatory, 3200 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_vExcitatory, dd_vExcitatory, 3200 * sizeof(scalar));
    cudaHostAlloc(&_regimeIDExcitatory, 3200 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d__regimeIDExcitatory, dd__regimeIDExcitatory, 3200 * sizeof(unsigned int));

    cudaHostAlloc(&glbSpkCntInhibitory, 2 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntInhibitory, dd_glbSpkCntInhibitory, 2 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkInhibitory, 1600 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkInhibitory, dd_glbSpkInhibitory, 1600 * sizeof(unsigned int));
    cudaHostAlloc(&t_spikeInhibitory, 800 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_t_spikeInhibitory, dd_t_spikeInhibitory, 800 * sizeof(scalar));
    cudaHostAlloc(&vInhibitory, 800 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_vInhibitory, dd_vInhibitory, 800 * sizeof(scalar));
    cudaHostAlloc(&_regimeIDInhibitory, 800 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d__regimeIDInhibitory, dd__regimeIDInhibitory, 800 * sizeof(unsigned int));

    cudaHostAlloc(&glbSpkCntSpike_Source, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntSpike_Source, dd_glbSpkCntSpike_Source, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkSpike_Source, 20 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkSpike_Source, dd_glbSpkSpike_Source, 20 * sizeof(unsigned int));

    cudaHostAlloc(&inSynExcitatory_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynExcitatory_to_Excitatory_Synapse_0_weight_update, dd_inSynExcitatory_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float));

    cudaHostAlloc(&inSynExcitatory_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update, dd_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float));

    cudaHostAlloc(&inSynInhibitory_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynInhibitory_to_Excitatory_Synapse_0_weight_update, dd_inSynInhibitory_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float));

    cudaHostAlloc(&inSynInhibitory_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update, dd_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float));

    cudaHostAlloc(&inSynSpike_Source_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update, dd_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float));

    cudaHostAlloc(&inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update, dd_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float));

}

void allocateExcitatory_to_Excitatory_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CExcitatory_to_Excitatory_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CExcitatory_to_Excitatory_Synapse_0_weight_update.indInG, 3201 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CExcitatory_to_Excitatory_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CExcitatory_to_Excitatory_Synapse_0_weight_update.preInd= NULL;
  CExcitatory_to_Excitatory_Synapse_0_weight_update.revIndInG= NULL;
  CExcitatory_to_Excitatory_Synapse_0_weight_update.revInd= NULL;
  CExcitatory_to_Excitatory_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGExcitatory_to_Excitatory_Synapse_0_weight_update, dd_indInGExcitatory_to_Excitatory_Synapse_0_weight_update, 3201 * sizeof(unsigned int));
    deviceMemAllocate(&d_indExcitatory_to_Excitatory_Synapse_0_weight_update, dd_indExcitatory_to_Excitatory_Synapse_0_weight_update, CExcitatory_to_Excitatory_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseExcitatory_to_Excitatory_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseExcitatory_to_Excitatory_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateExcitatory_to_Excitatory_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateExcitatory_to_Inhibitory_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CExcitatory_to_Inhibitory_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CExcitatory_to_Inhibitory_Synapse_0_weight_update.indInG, 3201 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CExcitatory_to_Inhibitory_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CExcitatory_to_Inhibitory_Synapse_0_weight_update.preInd= NULL;
  CExcitatory_to_Inhibitory_Synapse_0_weight_update.revIndInG= NULL;
  CExcitatory_to_Inhibitory_Synapse_0_weight_update.revInd= NULL;
  CExcitatory_to_Inhibitory_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGExcitatory_to_Inhibitory_Synapse_0_weight_update, dd_indInGExcitatory_to_Inhibitory_Synapse_0_weight_update, 3201 * sizeof(unsigned int));
    deviceMemAllocate(&d_indExcitatory_to_Inhibitory_Synapse_0_weight_update, dd_indExcitatory_to_Inhibitory_Synapse_0_weight_update, CExcitatory_to_Inhibitory_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseExcitatory_to_Inhibitory_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseExcitatory_to_Inhibitory_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateExcitatory_to_Inhibitory_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateInhibitory_to_Excitatory_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CInhibitory_to_Excitatory_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CInhibitory_to_Excitatory_Synapse_0_weight_update.indInG, 801 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CInhibitory_to_Excitatory_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CInhibitory_to_Excitatory_Synapse_0_weight_update.preInd= NULL;
  CInhibitory_to_Excitatory_Synapse_0_weight_update.revIndInG= NULL;
  CInhibitory_to_Excitatory_Synapse_0_weight_update.revInd= NULL;
  CInhibitory_to_Excitatory_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGInhibitory_to_Excitatory_Synapse_0_weight_update, dd_indInGInhibitory_to_Excitatory_Synapse_0_weight_update, 801 * sizeof(unsigned int));
    deviceMemAllocate(&d_indInhibitory_to_Excitatory_Synapse_0_weight_update, dd_indInhibitory_to_Excitatory_Synapse_0_weight_update, CInhibitory_to_Excitatory_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseInhibitory_to_Excitatory_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseInhibitory_to_Excitatory_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateInhibitory_to_Excitatory_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateInhibitory_to_Inhibitory_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CInhibitory_to_Inhibitory_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CInhibitory_to_Inhibitory_Synapse_0_weight_update.indInG, 801 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CInhibitory_to_Inhibitory_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CInhibitory_to_Inhibitory_Synapse_0_weight_update.preInd= NULL;
  CInhibitory_to_Inhibitory_Synapse_0_weight_update.revIndInG= NULL;
  CInhibitory_to_Inhibitory_Synapse_0_weight_update.revInd= NULL;
  CInhibitory_to_Inhibitory_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGInhibitory_to_Inhibitory_Synapse_0_weight_update, dd_indInGInhibitory_to_Inhibitory_Synapse_0_weight_update, 801 * sizeof(unsigned int));
    deviceMemAllocate(&d_indInhibitory_to_Inhibitory_Synapse_0_weight_update, dd_indInhibitory_to_Inhibitory_Synapse_0_weight_update, CInhibitory_to_Inhibitory_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseInhibitory_to_Inhibitory_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseInhibitory_to_Inhibitory_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateInhibitory_to_Inhibitory_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateSpike_Source_to_Excitatory_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CSpike_Source_to_Excitatory_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CSpike_Source_to_Excitatory_Synapse_0_weight_update.indInG, 21 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CSpike_Source_to_Excitatory_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CSpike_Source_to_Excitatory_Synapse_0_weight_update.preInd= NULL;
  CSpike_Source_to_Excitatory_Synapse_0_weight_update.revIndInG= NULL;
  CSpike_Source_to_Excitatory_Synapse_0_weight_update.revInd= NULL;
  CSpike_Source_to_Excitatory_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGSpike_Source_to_Excitatory_Synapse_0_weight_update, dd_indInGSpike_Source_to_Excitatory_Synapse_0_weight_update, 21 * sizeof(unsigned int));
    deviceMemAllocate(&d_indSpike_Source_to_Excitatory_Synapse_0_weight_update, dd_indSpike_Source_to_Excitatory_Synapse_0_weight_update, CSpike_Source_to_Excitatory_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseSpike_Source_to_Excitatory_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseSpike_Source_to_Excitatory_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateSpike_Source_to_Excitatory_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateSpike_Source_to_Inhibitory_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CSpike_Source_to_Inhibitory_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CSpike_Source_to_Inhibitory_Synapse_0_weight_update.indInG, 21 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CSpike_Source_to_Inhibitory_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CSpike_Source_to_Inhibitory_Synapse_0_weight_update.preInd= NULL;
  CSpike_Source_to_Inhibitory_Synapse_0_weight_update.revIndInG= NULL;
  CSpike_Source_to_Inhibitory_Synapse_0_weight_update.revInd= NULL;
  CSpike_Source_to_Inhibitory_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGSpike_Source_to_Inhibitory_Synapse_0_weight_update, dd_indInGSpike_Source_to_Inhibitory_Synapse_0_weight_update, 21 * sizeof(unsigned int));
    deviceMemAllocate(&d_indSpike_Source_to_Inhibitory_Synapse_0_weight_update, dd_indSpike_Source_to_Inhibitory_Synapse_0_weight_update, CSpike_Source_to_Inhibitory_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseSpike_Source_to_Inhibitory_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseSpike_Source_to_Inhibitory_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateSpike_Source_to_Inhibitory_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void freeMem()
{
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntExcitatory));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntExcitatory));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkExcitatory));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkExcitatory));
    CHECK_CUDA_ERRORS(cudaFreeHost(t_spikeExcitatory));
    CHECK_CUDA_ERRORS(cudaFree(d_t_spikeExcitatory));
    CHECK_CUDA_ERRORS(cudaFreeHost(vExcitatory));
    CHECK_CUDA_ERRORS(cudaFree(d_vExcitatory));
    CHECK_CUDA_ERRORS(cudaFreeHost(_regimeIDExcitatory));
    CHECK_CUDA_ERRORS(cudaFree(d__regimeIDExcitatory));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntInhibitory));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntInhibitory));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkInhibitory));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkInhibitory));
    CHECK_CUDA_ERRORS(cudaFreeHost(t_spikeInhibitory));
    CHECK_CUDA_ERRORS(cudaFree(d_t_spikeInhibitory));
    CHECK_CUDA_ERRORS(cudaFreeHost(vInhibitory));
    CHECK_CUDA_ERRORS(cudaFree(d_vInhibitory));
    CHECK_CUDA_ERRORS(cudaFreeHost(_regimeIDInhibitory));
    CHECK_CUDA_ERRORS(cudaFree(d__regimeIDInhibitory));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntSpike_Source));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntSpike_Source));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkSpike_Source));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkSpike_Source));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynExcitatory_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynExcitatory_to_Excitatory_Synapse_0_weight_update));
    CExcitatory_to_Excitatory_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CExcitatory_to_Excitatory_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGExcitatory_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CExcitatory_to_Excitatory_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indExcitatory_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynExcitatory_to_Inhibitory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update));
    CExcitatory_to_Inhibitory_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CExcitatory_to_Inhibitory_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGExcitatory_to_Inhibitory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CExcitatory_to_Inhibitory_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indExcitatory_to_Inhibitory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynInhibitory_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynInhibitory_to_Excitatory_Synapse_0_weight_update));
    CInhibitory_to_Excitatory_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CInhibitory_to_Excitatory_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGInhibitory_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CInhibitory_to_Excitatory_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indInhibitory_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynInhibitory_to_Inhibitory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update));
    CInhibitory_to_Inhibitory_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CInhibitory_to_Inhibitory_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGInhibitory_to_Inhibitory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CInhibitory_to_Inhibitory_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indInhibitory_to_Inhibitory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynSpike_Source_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update));
    CSpike_Source_to_Excitatory_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CSpike_Source_to_Excitatory_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGSpike_Source_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CSpike_Source_to_Excitatory_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indSpike_Source_to_Excitatory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update));
    CSpike_Source_to_Inhibitory_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CSpike_Source_to_Inhibitory_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGSpike_Source_to_Inhibitory_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CSpike_Source_to_Inhibitory_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indSpike_Source_to_Inhibitory_Synapse_0_weight_update));
}

void exitGeNN(){
  freeMem();
  cudaDeviceReset();
}

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)
void stepTimeCPU()
{
        calcSynapsesCPU(t);
    calcNeuronsCPU(t);
iT++;
t= iT*DT;
}
