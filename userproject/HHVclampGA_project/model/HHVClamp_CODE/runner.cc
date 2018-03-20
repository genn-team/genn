

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model HHVClamp containing general control code.
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

unsigned long long iT;
float t;


// ------------------------------------------------------------------------
// neuron variables

__device__ volatile unsigned int d_done;
unsigned int * glbSpkCntHH;
unsigned int * d_glbSpkCntHH;
__device__ unsigned int * dd_glbSpkCntHH;
unsigned int * glbSpkHH;
unsigned int * d_glbSpkHH;
__device__ unsigned int * dd_glbSpkHH;
scalar * VHH;
scalar * d_VHH;
__device__ scalar * dd_VHH;
scalar * mHH;
scalar * d_mHH;
__device__ scalar * dd_mHH;
scalar * hHH;
scalar * d_hHH;
__device__ scalar * dd_hHH;
scalar * nHH;
scalar * d_nHH;
__device__ scalar * dd_nHH;
scalar * gNaHH;
scalar * d_gNaHH;
__device__ scalar * dd_gNaHH;
scalar * ENaHH;
scalar * d_ENaHH;
__device__ scalar * dd_ENaHH;
scalar * gKHH;
scalar * d_gKHH;
__device__ scalar * dd_gKHH;
scalar * EKHH;
scalar * d_EKHH;
__device__ scalar * dd_EKHH;
scalar * glHH;
scalar * d_glHH;
__device__ scalar * dd_glHH;
scalar * ElHH;
scalar * d_ElHH;
__device__ scalar * dd_ElHH;
scalar * CHH;
scalar * d_CHH;
__device__ scalar * dd_CHH;
scalar * errHH;
scalar * d_errHH;
__device__ scalar * dd_errHH;
scalar stepVGHH;
scalar IsynGHH;

// ------------------------------------------------------------------------
// synapse variables


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
void allocateMem()
{
    CHECK_CUDA_ERRORS(cudaSetDevice(0));
    cudaHostAlloc(&glbSpkCntHH, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntHH, dd_glbSpkCntHH, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkHH, 12 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkHH, dd_glbSpkHH, 12 * sizeof(unsigned int));
    cudaHostAlloc(&VHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VHH, dd_VHH, 12 * sizeof(scalar));
    cudaHostAlloc(&mHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_mHH, dd_mHH, 12 * sizeof(scalar));
    cudaHostAlloc(&hHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_hHH, dd_hHH, 12 * sizeof(scalar));
    cudaHostAlloc(&nHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_nHH, dd_nHH, 12 * sizeof(scalar));
    cudaHostAlloc(&gNaHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gNaHH, dd_gNaHH, 12 * sizeof(scalar));
    cudaHostAlloc(&ENaHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_ENaHH, dd_ENaHH, 12 * sizeof(scalar));
    cudaHostAlloc(&gKHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gKHH, dd_gKHH, 12 * sizeof(scalar));
    cudaHostAlloc(&EKHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_EKHH, dd_EKHH, 12 * sizeof(scalar));
    cudaHostAlloc(&glHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_glHH, dd_glHH, 12 * sizeof(scalar));
    cudaHostAlloc(&ElHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_ElHH, dd_ElHH, 12 * sizeof(scalar));
    cudaHostAlloc(&CHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_CHH, dd_CHH, 12 * sizeof(scalar));
    cudaHostAlloc(&errHH, 12 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_errHH, dd_errHH, 12 * sizeof(scalar));

}

void freeMem()
{
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntHH));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkHH));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(VHH));
    CHECK_CUDA_ERRORS(cudaFree(d_VHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(mHH));
    CHECK_CUDA_ERRORS(cudaFree(d_mHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(hHH));
    CHECK_CUDA_ERRORS(cudaFree(d_hHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(nHH));
    CHECK_CUDA_ERRORS(cudaFree(d_nHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(gNaHH));
    CHECK_CUDA_ERRORS(cudaFree(d_gNaHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(ENaHH));
    CHECK_CUDA_ERRORS(cudaFree(d_ENaHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(gKHH));
    CHECK_CUDA_ERRORS(cudaFree(d_gKHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(EKHH));
    CHECK_CUDA_ERRORS(cudaFree(d_EKHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(glHH));
    CHECK_CUDA_ERRORS(cudaFree(d_glHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(ElHH));
    CHECK_CUDA_ERRORS(cudaFree(d_ElHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(CHH));
    CHECK_CUDA_ERRORS(cudaFree(d_CHH));
    CHECK_CUDA_ERRORS(cudaFreeHost(errHH));
    CHECK_CUDA_ERRORS(cudaFree(d_errHH));
}

void exitGeNN(){
  freeMem();
  cudaDeviceReset();
}

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)
void stepTimeCPU()
{
    calcNeuronsCPU(t);
iT++;
t= iT*DT;
}
