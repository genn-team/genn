

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model OneComp containing general control code.
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
unsigned int * glbSpkCntIzh1;
unsigned int * d_glbSpkCntIzh1;
__device__ unsigned int * dd_glbSpkCntIzh1;
unsigned int * glbSpkIzh1;
unsigned int * d_glbSpkIzh1;
__device__ unsigned int * dd_glbSpkIzh1;
scalar * VIzh1;
scalar * d_VIzh1;
__device__ scalar * dd_VIzh1;
scalar * UIzh1;
scalar * d_UIzh1;
__device__ scalar * dd_UIzh1;

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
    cudaHostAlloc(&glbSpkCntIzh1, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntIzh1, dd_glbSpkCntIzh1, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkIzh1, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkIzh1, dd_glbSpkIzh1, 1 * sizeof(unsigned int));
    cudaHostAlloc(&VIzh1, 1 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VIzh1, dd_VIzh1, 1 * sizeof(scalar));
    cudaHostAlloc(&UIzh1, 1 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_UIzh1, dd_UIzh1, 1 * sizeof(scalar));

}

void freeMem()
{
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntIzh1));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntIzh1));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkIzh1));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkIzh1));
    CHECK_CUDA_ERRORS(cudaFreeHost(VIzh1));
    CHECK_CUDA_ERRORS(cudaFree(d_VIzh1));
    CHECK_CUDA_ERRORS(cudaFreeHost(UIzh1));
    CHECK_CUDA_ERRORS(cudaFree(d_UIzh1));
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
