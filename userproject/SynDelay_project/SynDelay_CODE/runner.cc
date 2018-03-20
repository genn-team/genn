

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model SynDelay containing general control code.
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
unsigned int * glbSpkCntInput;
unsigned int * d_glbSpkCntInput;
__device__ unsigned int * dd_glbSpkCntInput;
unsigned int * glbSpkInput;
unsigned int * d_glbSpkInput;
__device__ unsigned int * dd_glbSpkInput;
unsigned int spkQuePtrInput;
__device__ volatile unsigned int dd_spkQuePtrInput;
scalar * VInput;
scalar * d_VInput;
__device__ scalar * dd_VInput;
scalar * UInput;
scalar * d_UInput;
__device__ scalar * dd_UInput;
unsigned int * glbSpkCntInter;
unsigned int * d_glbSpkCntInter;
__device__ unsigned int * dd_glbSpkCntInter;
unsigned int * glbSpkInter;
unsigned int * d_glbSpkInter;
__device__ unsigned int * dd_glbSpkInter;
scalar * VInter;
scalar * d_VInter;
__device__ scalar * dd_VInter;
scalar * UInter;
scalar * d_UInter;
__device__ scalar * dd_UInter;
unsigned int * glbSpkCntOutput;
unsigned int * d_glbSpkCntOutput;
__device__ unsigned int * dd_glbSpkCntOutput;
unsigned int * glbSpkOutput;
unsigned int * d_glbSpkOutput;
__device__ unsigned int * dd_glbSpkOutput;
scalar * VOutput;
scalar * d_VOutput;
__device__ scalar * dd_VOutput;
scalar * UOutput;
scalar * d_UOutput;
__device__ scalar * dd_UOutput;

// ------------------------------------------------------------------------
// synapse variables

float * inSynInputInter;
float * d_inSynInputInter;
__device__ float * dd_inSynInputInter;
float * inSynInputOutput;
float * d_inSynInputOutput;
__device__ float * dd_inSynInputOutput;
float * inSynInterOutput;
float * d_inSynInterOutput;
__device__ float * dd_inSynInterOutput;

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
    cudaHostAlloc(&glbSpkCntInput, 7 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntInput, dd_glbSpkCntInput, 7 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkInput, 3500 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkInput, dd_glbSpkInput, 3500 * sizeof(unsigned int));
    cudaHostAlloc(&VInput, 500 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VInput, dd_VInput, 500 * sizeof(scalar));
    cudaHostAlloc(&UInput, 500 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_UInput, dd_UInput, 500 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntInter, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntInter, dd_glbSpkCntInter, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkInter, 500 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkInter, dd_glbSpkInter, 500 * sizeof(unsigned int));
    cudaHostAlloc(&VInter, 500 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VInter, dd_VInter, 500 * sizeof(scalar));
    cudaHostAlloc(&UInter, 500 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_UInter, dd_UInter, 500 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntOutput, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntOutput, dd_glbSpkCntOutput, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkOutput, 500 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkOutput, dd_glbSpkOutput, 500 * sizeof(unsigned int));
    cudaHostAlloc(&VOutput, 500 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VOutput, dd_VOutput, 500 * sizeof(scalar));
    cudaHostAlloc(&UOutput, 500 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_UOutput, dd_UOutput, 500 * sizeof(scalar));

    cudaHostAlloc(&inSynInputInter, 500 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynInputInter, dd_inSynInputInter, 500 * sizeof(float));

    cudaHostAlloc(&inSynInputOutput, 500 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynInputOutput, dd_inSynInputOutput, 500 * sizeof(float));

    cudaHostAlloc(&inSynInterOutput, 500 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynInterOutput, dd_inSynInterOutput, 500 * sizeof(float));

}

void freeMem()
{
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntInput));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntInput));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkInput));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkInput));
    CHECK_CUDA_ERRORS(cudaFreeHost(VInput));
    CHECK_CUDA_ERRORS(cudaFree(d_VInput));
    CHECK_CUDA_ERRORS(cudaFreeHost(UInput));
    CHECK_CUDA_ERRORS(cudaFree(d_UInput));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntInter));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntInter));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkInter));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkInter));
    CHECK_CUDA_ERRORS(cudaFreeHost(VInter));
    CHECK_CUDA_ERRORS(cudaFree(d_VInter));
    CHECK_CUDA_ERRORS(cudaFreeHost(UInter));
    CHECK_CUDA_ERRORS(cudaFree(d_UInter));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntOutput));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntOutput));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkOutput));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkOutput));
    CHECK_CUDA_ERRORS(cudaFreeHost(VOutput));
    CHECK_CUDA_ERRORS(cudaFree(d_VOutput));
    CHECK_CUDA_ERRORS(cudaFreeHost(UOutput));
    CHECK_CUDA_ERRORS(cudaFree(d_UOutput));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynInputInter));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynInputInter));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynInputOutput));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynInputOutput));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynInterOutput));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynInterOutput));
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
