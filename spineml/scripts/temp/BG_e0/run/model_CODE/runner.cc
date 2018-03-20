

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


// ------------------------------------------------------------------------
// neuron variables

__device__ volatile unsigned int d_done;
unsigned int * glbSpkCntCortex;
unsigned int * d_glbSpkCntCortex;
__device__ unsigned int * dd_glbSpkCntCortex;
unsigned int * glbSpkCortex;
unsigned int * d_glbSpkCortex;
__device__ unsigned int * dd_glbSpkCortex;
scalar * aCortex;
scalar * d_aCortex;
__device__ scalar * dd_aCortex;
scalar * inCortex;
scalar * d_inCortex;
__device__ scalar * dd_inCortex;
scalar * outCortex;
scalar * d_outCortex;
__device__ scalar * dd_outCortex;
unsigned int * glbSpkCntD1;
unsigned int * d_glbSpkCntD1;
__device__ unsigned int * dd_glbSpkCntD1;
unsigned int * glbSpkD1;
unsigned int * d_glbSpkD1;
__device__ unsigned int * dd_glbSpkD1;
scalar * aD1;
scalar * d_aD1;
__device__ scalar * dd_aD1;
scalar * outD1;
scalar * d_outD1;
__device__ scalar * dd_outD1;
unsigned int * glbSpkCntD2;
unsigned int * d_glbSpkCntD2;
__device__ unsigned int * dd_glbSpkCntD2;
unsigned int * glbSpkD2;
unsigned int * d_glbSpkD2;
__device__ unsigned int * dd_glbSpkD2;
scalar * aD2;
scalar * d_aD2;
__device__ scalar * dd_aD2;
scalar * outD2;
scalar * d_outD2;
__device__ scalar * dd_outD2;
unsigned int * glbSpkCntGPe;
unsigned int * d_glbSpkCntGPe;
__device__ unsigned int * dd_glbSpkCntGPe;
unsigned int * glbSpkGPe;
unsigned int * d_glbSpkGPe;
__device__ unsigned int * dd_glbSpkGPe;
unsigned int spkQuePtrGPe;
__device__ volatile unsigned int dd_spkQuePtrGPe;
scalar * aGPe;
scalar * d_aGPe;
__device__ scalar * dd_aGPe;
scalar * outGPe;
scalar * d_outGPe;
__device__ scalar * dd_outGPe;
unsigned int * glbSpkCntGPi;
unsigned int * d_glbSpkCntGPi;
__device__ unsigned int * dd_glbSpkCntGPi;
unsigned int * glbSpkGPi;
unsigned int * d_glbSpkGPi;
__device__ unsigned int * dd_glbSpkGPi;
scalar * aGPi;
scalar * d_aGPi;
__device__ scalar * dd_aGPi;
scalar * outGPi;
scalar * d_outGPi;
__device__ scalar * dd_outGPi;
unsigned int * glbSpkCntSTN;
unsigned int * d_glbSpkCntSTN;
__device__ unsigned int * dd_glbSpkCntSTN;
unsigned int * glbSpkSTN;
unsigned int * d_glbSpkSTN;
__device__ unsigned int * dd_glbSpkSTN;
scalar * aSTN;
scalar * d_aSTN;
__device__ scalar * dd_aSTN;
scalar * outSTN;
scalar * d_outSTN;
__device__ scalar * dd_outSTN;

// ------------------------------------------------------------------------
// synapse variables

float * inSynCortex_to_D1_Synapse_0_weight_update;
float * d_inSynCortex_to_D1_Synapse_0_weight_update;
__device__ float * dd_inSynCortex_to_D1_Synapse_0_weight_update;
SparseProjection CCortex_to_D1_Synapse_0_weight_update;
unsigned int *d_indInGCortex_to_D1_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGCortex_to_D1_Synapse_0_weight_update;
unsigned int *d_indCortex_to_D1_Synapse_0_weight_update;
__device__ unsigned int *dd_indCortex_to_D1_Synapse_0_weight_update;
unsigned int *d_preIndCortex_to_D1_Synapse_0_weight_update;
__device__ unsigned int *dd_preIndCortex_to_D1_Synapse_0_weight_update;
float * inSynCortex_to_D2_Synapse_0_weight_update;
float * d_inSynCortex_to_D2_Synapse_0_weight_update;
__device__ float * dd_inSynCortex_to_D2_Synapse_0_weight_update;
SparseProjection CCortex_to_D2_Synapse_0_weight_update;
unsigned int *d_indInGCortex_to_D2_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGCortex_to_D2_Synapse_0_weight_update;
unsigned int *d_indCortex_to_D2_Synapse_0_weight_update;
__device__ unsigned int *dd_indCortex_to_D2_Synapse_0_weight_update;
unsigned int *d_preIndCortex_to_D2_Synapse_0_weight_update;
__device__ unsigned int *dd_preIndCortex_to_D2_Synapse_0_weight_update;
float * inSynCortex_to_STN_Synapse_0_weight_update;
float * d_inSynCortex_to_STN_Synapse_0_weight_update;
__device__ float * dd_inSynCortex_to_STN_Synapse_0_weight_update;
SparseProjection CCortex_to_STN_Synapse_0_weight_update;
unsigned int *d_indInGCortex_to_STN_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGCortex_to_STN_Synapse_0_weight_update;
unsigned int *d_indCortex_to_STN_Synapse_0_weight_update;
__device__ unsigned int *dd_indCortex_to_STN_Synapse_0_weight_update;
unsigned int *d_preIndCortex_to_STN_Synapse_0_weight_update;
__device__ unsigned int *dd_preIndCortex_to_STN_Synapse_0_weight_update;
float * inSynD1_to_GPi_Synapse_0_weight_update;
float * d_inSynD1_to_GPi_Synapse_0_weight_update;
__device__ float * dd_inSynD1_to_GPi_Synapse_0_weight_update;
SparseProjection CD1_to_GPi_Synapse_0_weight_update;
unsigned int *d_indInGD1_to_GPi_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGD1_to_GPi_Synapse_0_weight_update;
unsigned int *d_indD1_to_GPi_Synapse_0_weight_update;
__device__ unsigned int *dd_indD1_to_GPi_Synapse_0_weight_update;
unsigned int *d_preIndD1_to_GPi_Synapse_0_weight_update;
__device__ unsigned int *dd_preIndD1_to_GPi_Synapse_0_weight_update;
float * inSynD2_to_GPe_Synapse_0_weight_update;
float * d_inSynD2_to_GPe_Synapse_0_weight_update;
__device__ float * dd_inSynD2_to_GPe_Synapse_0_weight_update;
SparseProjection CD2_to_GPe_Synapse_0_weight_update;
unsigned int *d_indInGD2_to_GPe_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGD2_to_GPe_Synapse_0_weight_update;
unsigned int *d_indD2_to_GPe_Synapse_0_weight_update;
__device__ unsigned int *dd_indD2_to_GPe_Synapse_0_weight_update;
unsigned int *d_preIndD2_to_GPe_Synapse_0_weight_update;
__device__ unsigned int *dd_preIndD2_to_GPe_Synapse_0_weight_update;
float * inSynGPe_to_GPi_Synapse_0_weight_update;
float * d_inSynGPe_to_GPi_Synapse_0_weight_update;
__device__ float * dd_inSynGPe_to_GPi_Synapse_0_weight_update;
SparseProjection CGPe_to_GPi_Synapse_0_weight_update;
unsigned int *d_indInGGPe_to_GPi_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGGPe_to_GPi_Synapse_0_weight_update;
unsigned int *d_indGPe_to_GPi_Synapse_0_weight_update;
__device__ unsigned int *dd_indGPe_to_GPi_Synapse_0_weight_update;
unsigned int *d_preIndGPe_to_GPi_Synapse_0_weight_update;
__device__ unsigned int *dd_preIndGPe_to_GPi_Synapse_0_weight_update;
float * inSynGPe_to_STN_Synapse_0_weight_update;
float * d_inSynGPe_to_STN_Synapse_0_weight_update;
__device__ float * dd_inSynGPe_to_STN_Synapse_0_weight_update;
SparseProjection CGPe_to_STN_Synapse_0_weight_update;
unsigned int *d_indInGGPe_to_STN_Synapse_0_weight_update;
__device__ unsigned int *dd_indInGGPe_to_STN_Synapse_0_weight_update;
unsigned int *d_indGPe_to_STN_Synapse_0_weight_update;
__device__ unsigned int *dd_indGPe_to_STN_Synapse_0_weight_update;
unsigned int *d_preIndGPe_to_STN_Synapse_0_weight_update;
__device__ unsigned int *dd_preIndGPe_to_STN_Synapse_0_weight_update;
float * inSynSTN_to_GPe_Synapse_0_weight_update;
float * d_inSynSTN_to_GPe_Synapse_0_weight_update;
__device__ float * dd_inSynSTN_to_GPe_Synapse_0_weight_update;
float * inSynSTN_to_GPi_Synapse_0_weight_update;
float * d_inSynSTN_to_GPi_Synapse_0_weight_update;
__device__ float * dd_inSynSTN_to_GPi_Synapse_0_weight_update;

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
    cudaHostAlloc(&glbSpkCntCortex, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntCortex, dd_glbSpkCntCortex, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkCortex, 6 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCortex, dd_glbSpkCortex, 6 * sizeof(unsigned int));
    cudaHostAlloc(&aCortex, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_aCortex, dd_aCortex, 6 * sizeof(scalar));
    cudaHostAlloc(&inCortex, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_inCortex, dd_inCortex, 6 * sizeof(scalar));
    cudaHostAlloc(&outCortex, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_outCortex, dd_outCortex, 6 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntD1, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntD1, dd_glbSpkCntD1, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkD1, 6 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkD1, dd_glbSpkD1, 6 * sizeof(unsigned int));
    cudaHostAlloc(&aD1, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_aD1, dd_aD1, 6 * sizeof(scalar));
    cudaHostAlloc(&outD1, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_outD1, dd_outD1, 6 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntD2, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntD2, dd_glbSpkCntD2, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkD2, 6 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkD2, dd_glbSpkD2, 6 * sizeof(unsigned int));
    cudaHostAlloc(&aD2, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_aD2, dd_aD2, 6 * sizeof(scalar));
    cudaHostAlloc(&outD2, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_outD2, dd_outD2, 6 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntGPe, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntGPe, dd_glbSpkCntGPe, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkGPe, 6 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkGPe, dd_glbSpkGPe, 6 * sizeof(unsigned int));
    cudaHostAlloc(&aGPe, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_aGPe, dd_aGPe, 6 * sizeof(scalar));
    cudaHostAlloc(&outGPe, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_outGPe, dd_outGPe, 12 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntGPi, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntGPi, dd_glbSpkCntGPi, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkGPi, 6 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkGPi, dd_glbSpkGPi, 6 * sizeof(unsigned int));
    cudaHostAlloc(&aGPi, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_aGPi, dd_aGPi, 6 * sizeof(scalar));
    cudaHostAlloc(&outGPi, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_outGPi, dd_outGPi, 6 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntSTN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntSTN, dd_glbSpkCntSTN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkSTN, 6 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkSTN, dd_glbSpkSTN, 6 * sizeof(unsigned int));
    cudaHostAlloc(&aSTN, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_aSTN, dd_aSTN, 6 * sizeof(scalar));
    cudaHostAlloc(&outSTN, 6 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_outSTN, dd_outSTN, 6 * sizeof(scalar));

    cudaHostAlloc(&inSynCortex_to_D1_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynCortex_to_D1_Synapse_0_weight_update, dd_inSynCortex_to_D1_Synapse_0_weight_update, 6 * sizeof(float));

    cudaHostAlloc(&inSynCortex_to_D2_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynCortex_to_D2_Synapse_0_weight_update, dd_inSynCortex_to_D2_Synapse_0_weight_update, 6 * sizeof(float));

    cudaHostAlloc(&inSynCortex_to_STN_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynCortex_to_STN_Synapse_0_weight_update, dd_inSynCortex_to_STN_Synapse_0_weight_update, 6 * sizeof(float));

    cudaHostAlloc(&inSynD1_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynD1_to_GPi_Synapse_0_weight_update, dd_inSynD1_to_GPi_Synapse_0_weight_update, 6 * sizeof(float));

    cudaHostAlloc(&inSynD2_to_GPe_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynD2_to_GPe_Synapse_0_weight_update, dd_inSynD2_to_GPe_Synapse_0_weight_update, 6 * sizeof(float));

    cudaHostAlloc(&inSynGPe_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynGPe_to_GPi_Synapse_0_weight_update, dd_inSynGPe_to_GPi_Synapse_0_weight_update, 6 * sizeof(float));

    cudaHostAlloc(&inSynGPe_to_STN_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynGPe_to_STN_Synapse_0_weight_update, dd_inSynGPe_to_STN_Synapse_0_weight_update, 6 * sizeof(float));

    cudaHostAlloc(&inSynSTN_to_GPe_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynSTN_to_GPe_Synapse_0_weight_update, dd_inSynSTN_to_GPe_Synapse_0_weight_update, 6 * sizeof(float));

    cudaHostAlloc(&inSynSTN_to_GPi_Synapse_0_weight_update, 6 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynSTN_to_GPi_Synapse_0_weight_update, dd_inSynSTN_to_GPi_Synapse_0_weight_update, 6 * sizeof(float));

}

void allocateCortex_to_D1_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CCortex_to_D1_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CCortex_to_D1_Synapse_0_weight_update.indInG, 7 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CCortex_to_D1_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CCortex_to_D1_Synapse_0_weight_update.preInd, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CCortex_to_D1_Synapse_0_weight_update.revIndInG= NULL;
  CCortex_to_D1_Synapse_0_weight_update.revInd= NULL;
  CCortex_to_D1_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGCortex_to_D1_Synapse_0_weight_update, dd_indInGCortex_to_D1_Synapse_0_weight_update, 7 * sizeof(unsigned int));
    deviceMemAllocate(&d_indCortex_to_D1_Synapse_0_weight_update, dd_indCortex_to_D1_Synapse_0_weight_update, CCortex_to_D1_Synapse_0_weight_update.connN * sizeof(unsigned int));
    deviceMemAllocate(&d_preIndCortex_to_D1_Synapse_0_weight_update, dd_preIndCortex_to_D1_Synapse_0_weight_update, CCortex_to_D1_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseCortex_to_D1_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseCortex_to_D1_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateCortex_to_D1_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateCortex_to_D2_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CCortex_to_D2_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CCortex_to_D2_Synapse_0_weight_update.indInG, 7 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CCortex_to_D2_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CCortex_to_D2_Synapse_0_weight_update.preInd, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CCortex_to_D2_Synapse_0_weight_update.revIndInG= NULL;
  CCortex_to_D2_Synapse_0_weight_update.revInd= NULL;
  CCortex_to_D2_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGCortex_to_D2_Synapse_0_weight_update, dd_indInGCortex_to_D2_Synapse_0_weight_update, 7 * sizeof(unsigned int));
    deviceMemAllocate(&d_indCortex_to_D2_Synapse_0_weight_update, dd_indCortex_to_D2_Synapse_0_weight_update, CCortex_to_D2_Synapse_0_weight_update.connN * sizeof(unsigned int));
    deviceMemAllocate(&d_preIndCortex_to_D2_Synapse_0_weight_update, dd_preIndCortex_to_D2_Synapse_0_weight_update, CCortex_to_D2_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseCortex_to_D2_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseCortex_to_D2_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateCortex_to_D2_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateCortex_to_STN_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CCortex_to_STN_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CCortex_to_STN_Synapse_0_weight_update.indInG, 7 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CCortex_to_STN_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CCortex_to_STN_Synapse_0_weight_update.preInd, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CCortex_to_STN_Synapse_0_weight_update.revIndInG= NULL;
  CCortex_to_STN_Synapse_0_weight_update.revInd= NULL;
  CCortex_to_STN_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGCortex_to_STN_Synapse_0_weight_update, dd_indInGCortex_to_STN_Synapse_0_weight_update, 7 * sizeof(unsigned int));
    deviceMemAllocate(&d_indCortex_to_STN_Synapse_0_weight_update, dd_indCortex_to_STN_Synapse_0_weight_update, CCortex_to_STN_Synapse_0_weight_update.connN * sizeof(unsigned int));
    deviceMemAllocate(&d_preIndCortex_to_STN_Synapse_0_weight_update, dd_preIndCortex_to_STN_Synapse_0_weight_update, CCortex_to_STN_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseCortex_to_STN_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseCortex_to_STN_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateCortex_to_STN_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateD1_to_GPi_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CD1_to_GPi_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CD1_to_GPi_Synapse_0_weight_update.indInG, 7 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CD1_to_GPi_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CD1_to_GPi_Synapse_0_weight_update.preInd, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CD1_to_GPi_Synapse_0_weight_update.revIndInG= NULL;
  CD1_to_GPi_Synapse_0_weight_update.revInd= NULL;
  CD1_to_GPi_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGD1_to_GPi_Synapse_0_weight_update, dd_indInGD1_to_GPi_Synapse_0_weight_update, 7 * sizeof(unsigned int));
    deviceMemAllocate(&d_indD1_to_GPi_Synapse_0_weight_update, dd_indD1_to_GPi_Synapse_0_weight_update, CD1_to_GPi_Synapse_0_weight_update.connN * sizeof(unsigned int));
    deviceMemAllocate(&d_preIndD1_to_GPi_Synapse_0_weight_update, dd_preIndD1_to_GPi_Synapse_0_weight_update, CD1_to_GPi_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseD1_to_GPi_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseD1_to_GPi_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateD1_to_GPi_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateD2_to_GPe_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CD2_to_GPe_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CD2_to_GPe_Synapse_0_weight_update.indInG, 7 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CD2_to_GPe_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CD2_to_GPe_Synapse_0_weight_update.preInd, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CD2_to_GPe_Synapse_0_weight_update.revIndInG= NULL;
  CD2_to_GPe_Synapse_0_weight_update.revInd= NULL;
  CD2_to_GPe_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGD2_to_GPe_Synapse_0_weight_update, dd_indInGD2_to_GPe_Synapse_0_weight_update, 7 * sizeof(unsigned int));
    deviceMemAllocate(&d_indD2_to_GPe_Synapse_0_weight_update, dd_indD2_to_GPe_Synapse_0_weight_update, CD2_to_GPe_Synapse_0_weight_update.connN * sizeof(unsigned int));
    deviceMemAllocate(&d_preIndD2_to_GPe_Synapse_0_weight_update, dd_preIndD2_to_GPe_Synapse_0_weight_update, CD2_to_GPe_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseD2_to_GPe_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseD2_to_GPe_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateD2_to_GPe_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateGPe_to_GPi_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CGPe_to_GPi_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CGPe_to_GPi_Synapse_0_weight_update.indInG, 7 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CGPe_to_GPi_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CGPe_to_GPi_Synapse_0_weight_update.preInd, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CGPe_to_GPi_Synapse_0_weight_update.revIndInG= NULL;
  CGPe_to_GPi_Synapse_0_weight_update.revInd= NULL;
  CGPe_to_GPi_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGGPe_to_GPi_Synapse_0_weight_update, dd_indInGGPe_to_GPi_Synapse_0_weight_update, 7 * sizeof(unsigned int));
    deviceMemAllocate(&d_indGPe_to_GPi_Synapse_0_weight_update, dd_indGPe_to_GPi_Synapse_0_weight_update, CGPe_to_GPi_Synapse_0_weight_update.connN * sizeof(unsigned int));
    deviceMemAllocate(&d_preIndGPe_to_GPi_Synapse_0_weight_update, dd_preIndGPe_to_GPi_Synapse_0_weight_update, CGPe_to_GPi_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseGPe_to_GPi_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseGPe_to_GPi_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateGPe_to_GPi_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateGPe_to_STN_Synapse_0_weight_update(unsigned int connN){
// Allocate host side variables
  CGPe_to_STN_Synapse_0_weight_update.connN= connN;
    cudaHostAlloc(&CGPe_to_STN_Synapse_0_weight_update.indInG, 7 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CGPe_to_STN_Synapse_0_weight_update.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CGPe_to_STN_Synapse_0_weight_update.preInd, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CGPe_to_STN_Synapse_0_weight_update.revIndInG= NULL;
  CGPe_to_STN_Synapse_0_weight_update.revInd= NULL;
  CGPe_to_STN_Synapse_0_weight_update.remap= NULL;
    deviceMemAllocate(&d_indInGGPe_to_STN_Synapse_0_weight_update, dd_indInGGPe_to_STN_Synapse_0_weight_update, 7 * sizeof(unsigned int));
    deviceMemAllocate(&d_indGPe_to_STN_Synapse_0_weight_update, dd_indGPe_to_STN_Synapse_0_weight_update, CGPe_to_STN_Synapse_0_weight_update.connN * sizeof(unsigned int));
    deviceMemAllocate(&d_preIndGPe_to_STN_Synapse_0_weight_update, dd_preIndGPe_to_STN_Synapse_0_weight_update, CGPe_to_STN_Synapse_0_weight_update.connN * sizeof(unsigned int));
}

void createSparseConnectivityFromDenseGPe_to_STN_Synapse_0_weight_update(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseGPe_to_STN_Synapse_0_weight_update() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateGPe_to_STN_Synapse_0_weight_update(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void freeMem()
{
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntCortex));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntCortex));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCortex));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCortex));
    CHECK_CUDA_ERRORS(cudaFreeHost(aCortex));
    CHECK_CUDA_ERRORS(cudaFree(d_aCortex));
    CHECK_CUDA_ERRORS(cudaFreeHost(inCortex));
    CHECK_CUDA_ERRORS(cudaFree(d_inCortex));
    CHECK_CUDA_ERRORS(cudaFreeHost(outCortex));
    CHECK_CUDA_ERRORS(cudaFree(d_outCortex));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntD1));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntD1));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkD1));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkD1));
    CHECK_CUDA_ERRORS(cudaFreeHost(aD1));
    CHECK_CUDA_ERRORS(cudaFree(d_aD1));
    CHECK_CUDA_ERRORS(cudaFreeHost(outD1));
    CHECK_CUDA_ERRORS(cudaFree(d_outD1));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntD2));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntD2));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkD2));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkD2));
    CHECK_CUDA_ERRORS(cudaFreeHost(aD2));
    CHECK_CUDA_ERRORS(cudaFree(d_aD2));
    CHECK_CUDA_ERRORS(cudaFreeHost(outD2));
    CHECK_CUDA_ERRORS(cudaFree(d_outD2));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntGPe));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntGPe));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkGPe));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkGPe));
    CHECK_CUDA_ERRORS(cudaFreeHost(aGPe));
    CHECK_CUDA_ERRORS(cudaFree(d_aGPe));
    CHECK_CUDA_ERRORS(cudaFreeHost(outGPe));
    CHECK_CUDA_ERRORS(cudaFree(d_outGPe));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntGPi));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntGPi));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkGPi));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkGPi));
    CHECK_CUDA_ERRORS(cudaFreeHost(aGPi));
    CHECK_CUDA_ERRORS(cudaFree(d_aGPi));
    CHECK_CUDA_ERRORS(cudaFreeHost(outGPi));
    CHECK_CUDA_ERRORS(cudaFree(d_outGPi));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntSTN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntSTN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkSTN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkSTN));
    CHECK_CUDA_ERRORS(cudaFreeHost(aSTN));
    CHECK_CUDA_ERRORS(cudaFree(d_aSTN));
    CHECK_CUDA_ERRORS(cudaFreeHost(outSTN));
    CHECK_CUDA_ERRORS(cudaFree(d_outSTN));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynCortex_to_D1_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynCortex_to_D1_Synapse_0_weight_update));
    CCortex_to_D1_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_D1_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGCortex_to_D1_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_D1_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indCortex_to_D1_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_D1_Synapse_0_weight_update.preInd));
    CHECK_CUDA_ERRORS(cudaFree(d_preIndCortex_to_D1_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynCortex_to_D2_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynCortex_to_D2_Synapse_0_weight_update));
    CCortex_to_D2_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_D2_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGCortex_to_D2_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_D2_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indCortex_to_D2_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_D2_Synapse_0_weight_update.preInd));
    CHECK_CUDA_ERRORS(cudaFree(d_preIndCortex_to_D2_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynCortex_to_STN_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynCortex_to_STN_Synapse_0_weight_update));
    CCortex_to_STN_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_STN_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGCortex_to_STN_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_STN_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indCortex_to_STN_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CCortex_to_STN_Synapse_0_weight_update.preInd));
    CHECK_CUDA_ERRORS(cudaFree(d_preIndCortex_to_STN_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynD1_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynD1_to_GPi_Synapse_0_weight_update));
    CD1_to_GPi_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CD1_to_GPi_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGD1_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CD1_to_GPi_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indD1_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CD1_to_GPi_Synapse_0_weight_update.preInd));
    CHECK_CUDA_ERRORS(cudaFree(d_preIndD1_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynD2_to_GPe_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynD2_to_GPe_Synapse_0_weight_update));
    CD2_to_GPe_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CD2_to_GPe_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGD2_to_GPe_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CD2_to_GPe_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indD2_to_GPe_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CD2_to_GPe_Synapse_0_weight_update.preInd));
    CHECK_CUDA_ERRORS(cudaFree(d_preIndD2_to_GPe_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynGPe_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynGPe_to_GPi_Synapse_0_weight_update));
    CGPe_to_GPi_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CGPe_to_GPi_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGGPe_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CGPe_to_GPi_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indGPe_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CGPe_to_GPi_Synapse_0_weight_update.preInd));
    CHECK_CUDA_ERRORS(cudaFree(d_preIndGPe_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynGPe_to_STN_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynGPe_to_STN_Synapse_0_weight_update));
    CGPe_to_STN_Synapse_0_weight_update.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CGPe_to_STN_Synapse_0_weight_update.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGGPe_to_STN_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CGPe_to_STN_Synapse_0_weight_update.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indGPe_to_STN_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(CGPe_to_STN_Synapse_0_weight_update.preInd));
    CHECK_CUDA_ERRORS(cudaFree(d_preIndGPe_to_STN_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynSTN_to_GPe_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynSTN_to_GPe_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynSTN_to_GPi_Synapse_0_weight_update));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynSTN_to_GPi_Synapse_0_weight_update));
}

void exitGeNN(){
  freeMem();
  cudaDeviceReset();
}

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)
void stepTimeCPU()
{
        calcSynapseDynamicsCPU(t);
        calcSynapsesCPU(t);
    calcNeuronsCPU(t);
iT++;
t= iT*DT;
}
