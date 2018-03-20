

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model MBody1 containing general control code.
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
unsigned int * glbSpkCntDN;
unsigned int * d_glbSpkCntDN;
__device__ unsigned int * dd_glbSpkCntDN;
unsigned int * glbSpkDN;
unsigned int * d_glbSpkDN;
__device__ unsigned int * dd_glbSpkDN;
unsigned int * glbSpkCntEvntDN;
unsigned int * d_glbSpkCntEvntDN;
__device__ unsigned int * dd_glbSpkCntEvntDN;
unsigned int * glbSpkEvntDN;
unsigned int * d_glbSpkEvntDN;
__device__ unsigned int * dd_glbSpkEvntDN;
float * sTDN;
float * d_sTDN;
__device__ float * dd_sTDN;
scalar * VDN;
scalar * d_VDN;
__device__ scalar * dd_VDN;
scalar * mDN;
scalar * d_mDN;
__device__ scalar * dd_mDN;
scalar * hDN;
scalar * d_hDN;
__device__ scalar * dd_hDN;
scalar * nDN;
scalar * d_nDN;
__device__ scalar * dd_nDN;
unsigned int * glbSpkCntKC;
unsigned int * d_glbSpkCntKC;
__device__ unsigned int * dd_glbSpkCntKC;
unsigned int * glbSpkKC;
unsigned int * d_glbSpkKC;
__device__ unsigned int * dd_glbSpkKC;
float * sTKC;
float * d_sTKC;
__device__ float * dd_sTKC;
scalar * VKC;
scalar * d_VKC;
__device__ scalar * dd_VKC;
scalar * mKC;
scalar * d_mKC;
__device__ scalar * dd_mKC;
scalar * hKC;
scalar * d_hKC;
__device__ scalar * dd_hKC;
scalar * nKC;
scalar * d_nKC;
__device__ scalar * dd_nKC;
unsigned int * glbSpkCntLHI;
unsigned int * d_glbSpkCntLHI;
__device__ unsigned int * dd_glbSpkCntLHI;
unsigned int * glbSpkLHI;
unsigned int * d_glbSpkLHI;
__device__ unsigned int * dd_glbSpkLHI;
unsigned int * glbSpkCntEvntLHI;
unsigned int * d_glbSpkCntEvntLHI;
__device__ unsigned int * dd_glbSpkCntEvntLHI;
unsigned int * glbSpkEvntLHI;
unsigned int * d_glbSpkEvntLHI;
__device__ unsigned int * dd_glbSpkEvntLHI;
scalar * VLHI;
scalar * d_VLHI;
__device__ scalar * dd_VLHI;
scalar * mLHI;
scalar * d_mLHI;
__device__ scalar * dd_mLHI;
scalar * hLHI;
scalar * d_hLHI;
__device__ scalar * dd_hLHI;
scalar * nLHI;
scalar * d_nLHI;
__device__ scalar * dd_nLHI;
unsigned int * glbSpkCntPN;
unsigned int * d_glbSpkCntPN;
__device__ unsigned int * dd_glbSpkCntPN;
unsigned int * glbSpkPN;
unsigned int * d_glbSpkPN;
__device__ unsigned int * dd_glbSpkPN;
scalar * VPN;
scalar * d_VPN;
__device__ scalar * dd_VPN;
uint64_t * seedPN;
uint64_t * d_seedPN;
__device__ uint64_t * dd_seedPN;
scalar * spikeTimePN;
scalar * d_spikeTimePN;
__device__ scalar * dd_spikeTimePN;
uint64_t * ratesPN;
unsigned int offsetPN;

// ------------------------------------------------------------------------
// synapse variables

float * inSynDNDN;
float * d_inSynDNDN;
__device__ float * dd_inSynDNDN;
float * inSynKCDN;
float * d_inSynKCDN;
__device__ float * dd_inSynKCDN;
scalar * gKCDN;
scalar * d_gKCDN;
__device__ scalar * dd_gKCDN;
scalar * gRawKCDN;
scalar * d_gRawKCDN;
__device__ scalar * dd_gRawKCDN;
float * inSynLHIKC;
float * d_inSynLHIKC;
__device__ float * dd_inSynLHIKC;
float * inSynPNKC;
float * d_inSynPNKC;
__device__ float * dd_inSynPNKC;
scalar * gPNKC;
scalar * d_gPNKC;
__device__ scalar * dd_gPNKC;
float * inSynPNLHI;
float * d_inSynPNLHI;
__device__ float * dd_inSynPNLHI;
scalar * gPNLHI;
scalar * d_gPNLHI;
__device__ scalar * dd_gPNLHI;

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
    cudaHostAlloc(&glbSpkCntDN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntDN, dd_glbSpkCntDN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkDN, 100 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkDN, dd_glbSpkDN, 100 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkCntEvntDN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntEvntDN, dd_glbSpkCntEvntDN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkEvntDN, 100 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkEvntDN, dd_glbSpkEvntDN, 100 * sizeof(unsigned int));
    cudaHostAlloc(&sTDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_sTDN, dd_sTDN, 100 * sizeof(float));
    cudaHostAlloc(&VDN, 100 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VDN, dd_VDN, 100 * sizeof(scalar));
    cudaHostAlloc(&mDN, 100 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_mDN, dd_mDN, 100 * sizeof(scalar));
    cudaHostAlloc(&hDN, 100 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_hDN, dd_hDN, 100 * sizeof(scalar));
    cudaHostAlloc(&nDN, 100 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_nDN, dd_nDN, 100 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntKC, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntKC, dd_glbSpkCntKC, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkKC, 1000 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkKC, dd_glbSpkKC, 1000 * sizeof(unsigned int));
    cudaHostAlloc(&sTKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_sTKC, dd_sTKC, 1000 * sizeof(float));
    cudaHostAlloc(&VKC, 1000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VKC, dd_VKC, 1000 * sizeof(scalar));
    cudaHostAlloc(&mKC, 1000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_mKC, dd_mKC, 1000 * sizeof(scalar));
    cudaHostAlloc(&hKC, 1000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_hKC, dd_hKC, 1000 * sizeof(scalar));
    cudaHostAlloc(&nKC, 1000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_nKC, dd_nKC, 1000 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntLHI, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntLHI, dd_glbSpkCntLHI, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkLHI, 20 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkLHI, dd_glbSpkLHI, 20 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkCntEvntLHI, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntEvntLHI, dd_glbSpkCntEvntLHI, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkEvntLHI, 20 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkEvntLHI, dd_glbSpkEvntLHI, 20 * sizeof(unsigned int));
    cudaHostAlloc(&VLHI, 20 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VLHI, dd_VLHI, 20 * sizeof(scalar));
    cudaHostAlloc(&mLHI, 20 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_mLHI, dd_mLHI, 20 * sizeof(scalar));
    cudaHostAlloc(&hLHI, 20 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_hLHI, dd_hLHI, 20 * sizeof(scalar));
    cudaHostAlloc(&nLHI, 20 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_nLHI, dd_nLHI, 20 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntPN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntPN, dd_glbSpkCntPN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkPN, 100 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkPN, dd_glbSpkPN, 100 * sizeof(unsigned int));
    cudaHostAlloc(&VPN, 100 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VPN, dd_VPN, 100 * sizeof(scalar));
    cudaHostAlloc(&seedPN, 100 * sizeof(uint64_t), cudaHostAllocPortable);
    deviceMemAllocate(&d_seedPN, dd_seedPN, 100 * sizeof(uint64_t));
    cudaHostAlloc(&spikeTimePN, 100 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_spikeTimePN, dd_spikeTimePN, 100 * sizeof(scalar));

    cudaHostAlloc(&inSynDNDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynDNDN, dd_inSynDNDN, 100 * sizeof(float));

    cudaHostAlloc(&inSynKCDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynKCDN, dd_inSynKCDN, 100 * sizeof(float));
    cudaHostAlloc(&gKCDN, 100000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gKCDN, dd_gKCDN, 100000 * sizeof(scalar));
    cudaHostAlloc(&gRawKCDN, 100000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gRawKCDN, dd_gRawKCDN, 100000 * sizeof(scalar));

    cudaHostAlloc(&inSynLHIKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynLHIKC, dd_inSynLHIKC, 1000 * sizeof(float));

    cudaHostAlloc(&inSynPNKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynPNKC, dd_inSynPNKC, 1000 * sizeof(float));
    cudaHostAlloc(&gPNKC, 100000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gPNKC, dd_gPNKC, 100000 * sizeof(scalar));

    cudaHostAlloc(&inSynPNLHI, 20 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynPNLHI, dd_inSynPNLHI, 20 * sizeof(float));
    cudaHostAlloc(&gPNLHI, 2000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gPNLHI, dd_gPNLHI, 2000 * sizeof(scalar));

}

void freeMem()
{
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntDN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkDN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntEvntDN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvntDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkEvntDN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvntDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(sTDN));
    CHECK_CUDA_ERRORS(cudaFree(d_sTDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(VDN));
    CHECK_CUDA_ERRORS(cudaFree(d_VDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(mDN));
    CHECK_CUDA_ERRORS(cudaFree(d_mDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(hDN));
    CHECK_CUDA_ERRORS(cudaFree(d_hDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(nDN));
    CHECK_CUDA_ERRORS(cudaFree(d_nDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntKC));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkKC));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(sTKC));
    CHECK_CUDA_ERRORS(cudaFree(d_sTKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(VKC));
    CHECK_CUDA_ERRORS(cudaFree(d_VKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(mKC));
    CHECK_CUDA_ERRORS(cudaFree(d_mKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(hKC));
    CHECK_CUDA_ERRORS(cudaFree(d_hKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(nKC));
    CHECK_CUDA_ERRORS(cudaFree(d_nKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntEvntLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvntLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkEvntLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvntLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(VLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_VLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(mLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_mLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(hLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_hLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(nLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_nLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntPN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkPN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(VPN));
    CHECK_CUDA_ERRORS(cudaFree(d_VPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(seedPN));
    CHECK_CUDA_ERRORS(cudaFree(d_seedPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(spikeTimePN));
    CHECK_CUDA_ERRORS(cudaFree(d_spikeTimePN));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynDNDN));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynDNDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynKCDN));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynKCDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(gKCDN));
    CHECK_CUDA_ERRORS(cudaFree(d_gKCDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(gRawKCDN));
    CHECK_CUDA_ERRORS(cudaFree(d_gRawKCDN));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynLHIKC));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynLHIKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynPNKC));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynPNKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(gPNKC));
    CHECK_CUDA_ERRORS(cudaFree(d_gPNKC));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynPNLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynPNLHI));
    CHECK_CUDA_ERRORS(cudaFreeHost(gPNLHI));
    CHECK_CUDA_ERRORS(cudaFree(d_gPNLHI));
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
        learnSynapsesPostHost(t);
    calcNeuronsCPU(t);
iT++;
t= iT*DT;
}
