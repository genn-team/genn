

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model Schmuker_2014_classifier containing general control code.
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

// ------------------------------------------------------------------------
// global variables

unsigned long long iT= 0;
float t;

// ------------------------------------------------------------------------
// neuron variables

__device__ volatile unsigned int d_done;
unsigned int * glbSpkCntAN;
unsigned int * d_glbSpkCntAN;
__device__ unsigned int * dd_glbSpkCntAN;
unsigned int * glbSpkAN;
unsigned int * d_glbSpkAN;
__device__ unsigned int * dd_glbSpkAN;
unsigned int * glbSpkCntEvntAN;
unsigned int * d_glbSpkCntEvntAN;
__device__ unsigned int * dd_glbSpkCntEvntAN;
unsigned int * glbSpkEvntAN;
unsigned int * d_glbSpkEvntAN;
__device__ unsigned int * dd_glbSpkEvntAN;
scalar * VAN;
scalar * d_VAN;
__device__ scalar * dd_VAN;
scalar * preVAN;
scalar * d_preVAN;
__device__ scalar * dd_preVAN;
unsigned int * glbSpkCntPN;
unsigned int * d_glbSpkCntPN;
__device__ unsigned int * dd_glbSpkCntPN;
unsigned int * glbSpkPN;
unsigned int * d_glbSpkPN;
__device__ unsigned int * dd_glbSpkPN;
unsigned int * glbSpkCntEvntPN;
unsigned int * d_glbSpkCntEvntPN;
__device__ unsigned int * dd_glbSpkCntEvntPN;
unsigned int * glbSpkEvntPN;
unsigned int * d_glbSpkEvntPN;
__device__ unsigned int * dd_glbSpkEvntPN;
scalar * VPN;
scalar * d_VPN;
__device__ scalar * dd_VPN;
scalar * preVPN;
scalar * d_preVPN;
__device__ scalar * dd_preVPN;
unsigned int * glbSpkCntRN;
unsigned int * d_glbSpkCntRN;
__device__ unsigned int * dd_glbSpkCntRN;
unsigned int * glbSpkRN;
unsigned int * d_glbSpkRN;
__device__ unsigned int * dd_glbSpkRN;
unsigned int * glbSpkCntEvntRN;
unsigned int * d_glbSpkCntEvntRN;
__device__ unsigned int * dd_glbSpkCntEvntRN;
unsigned int * glbSpkEvntRN;
unsigned int * d_glbSpkEvntRN;
__device__ unsigned int * dd_glbSpkEvntRN;
scalar * VRN;
scalar * d_VRN;
__device__ scalar * dd_VRN;
uint64_t * seedRN;
uint64_t * d_seedRN;
__device__ uint64_t * dd_seedRN;
scalar * spikeTimeRN;
scalar * d_spikeTimeRN;
__device__ scalar * dd_spikeTimeRN;
uint64_t * ratesRN;
unsigned int offsetRN;

// ------------------------------------------------------------------------
// synapse variables

float * inSynANAN;
float * d_inSynANAN;
__device__ float * dd_inSynANAN;
scalar * gANAN;
scalar * d_gANAN;
__device__ scalar * dd_gANAN;
float * inSynPNAN;
float * d_inSynPNAN;
__device__ float * dd_inSynPNAN;
scalar * gPNAN;
scalar * d_gPNAN;
__device__ scalar * dd_gPNAN;
float * inSynPNPN;
float * d_inSynPNPN;
__device__ float * dd_inSynPNPN;
scalar * gPNPN;
scalar * d_gPNPN;
__device__ scalar * dd_gPNPN;
float * inSynRNPN;
float * d_inSynRNPN;
__device__ float * dd_inSynRNPN;
scalar * gRNPN;
scalar * d_gRNPN;
__device__ scalar * dd_gRNPN;

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

#include "neuronFnct.cc"
#include "synapseFnct.cc"
void allocateMem()
{
    CHECK_CUDA_ERRORS(cudaSetDevice(0));
    cudaHostAlloc(&glbSpkCntAN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntAN, dd_glbSpkCntAN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkAN, 180 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkAN, dd_glbSpkAN, 180 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkCntEvntAN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntEvntAN, dd_glbSpkCntEvntAN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkEvntAN, 180 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkEvntAN, dd_glbSpkEvntAN, 180 * sizeof(unsigned int));
    cudaHostAlloc(&VAN, 180 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VAN, dd_VAN, 180 * sizeof(scalar));
    cudaHostAlloc(&preVAN, 180 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_preVAN, dd_preVAN, 180 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntPN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntPN, dd_glbSpkCntPN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkPN, 600 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkPN, dd_glbSpkPN, 600 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkCntEvntPN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntEvntPN, dd_glbSpkCntEvntPN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkEvntPN, 600 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkEvntPN, dd_glbSpkEvntPN, 600 * sizeof(unsigned int));
    cudaHostAlloc(&VPN, 600 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VPN, dd_VPN, 600 * sizeof(scalar));
    cudaHostAlloc(&preVPN, 600 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_preVPN, dd_preVPN, 600 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntRN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntRN, dd_glbSpkCntRN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkRN, 600 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkRN, dd_glbSpkRN, 600 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkCntEvntRN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntEvntRN, dd_glbSpkCntEvntRN, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkEvntRN, 600 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkEvntRN, dd_glbSpkEvntRN, 600 * sizeof(unsigned int));
    cudaHostAlloc(&VRN, 600 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VRN, dd_VRN, 600 * sizeof(scalar));
    cudaHostAlloc(&seedRN, 600 * sizeof(uint64_t), cudaHostAllocPortable);
    deviceMemAllocate(&d_seedRN, dd_seedRN, 600 * sizeof(uint64_t));
    cudaHostAlloc(&spikeTimeRN, 600 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_spikeTimeRN, dd_spikeTimeRN, 600 * sizeof(scalar));

    cudaHostAlloc(&inSynANAN, 180 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynANAN, dd_inSynANAN, 180 * sizeof(float));
    cudaHostAlloc(&gANAN, 32400 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gANAN, dd_gANAN, 32400 * sizeof(scalar));

    cudaHostAlloc(&inSynPNAN, 180 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynPNAN, dd_inSynPNAN, 180 * sizeof(float));
    cudaHostAlloc(&gPNAN, 108000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gPNAN, dd_gPNAN, 108000 * sizeof(scalar));

    cudaHostAlloc(&inSynPNPN, 600 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynPNPN, dd_inSynPNPN, 600 * sizeof(float));
    cudaHostAlloc(&gPNPN, 360000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gPNPN, dd_gPNPN, 360000 * sizeof(scalar));

    cudaHostAlloc(&inSynRNPN, 600 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynRNPN, dd_inSynRNPN, 600 * sizeof(float));
    cudaHostAlloc(&gRNPN, 360000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gRNPN, dd_gRNPN, 360000 * sizeof(scalar));

}

//-------------------------------------------------------------------------
/*! \brief Function to (re)set all model variables to their compile-time, homogeneous initial values.
 Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device.
*/
//-------------------------------------------------------------------------

void initialize()
{
    srand((unsigned int) time(NULL));

    // neuron variables
    glbSpkCntAN[0] = 0;
    for (int i = 0; i < 180; i++) {
        glbSpkAN[i] = 0;
    }
    glbSpkCntEvntAN[0] = 0;
    for (int i = 0; i < 180; i++) {
        glbSpkEvntAN[i] = 0;
    }
    for (int i = 0; i < 180; i++) {
        VAN[i] = -60;
    }
    for (int i = 0; i < 180; i++) {
        preVAN[i] = -60;
    }
    glbSpkCntPN[0] = 0;
    for (int i = 0; i < 600; i++) {
        glbSpkPN[i] = 0;
    }
    glbSpkCntEvntPN[0] = 0;
    for (int i = 0; i < 600; i++) {
        glbSpkEvntPN[i] = 0;
    }
    for (int i = 0; i < 600; i++) {
        VPN[i] = -60;
    }
    for (int i = 0; i < 600; i++) {
        preVPN[i] = -60;
    }
    glbSpkCntRN[0] = 0;
    for (int i = 0; i < 600; i++) {
        glbSpkRN[i] = 0;
    }
    glbSpkCntEvntRN[0] = 0;
    for (int i = 0; i < 600; i++) {
        glbSpkEvntRN[i] = 0;
    }
    for (int i = 0; i < 600; i++) {
        VRN[i] = -60;
    }
    for (int i = 0; i < 600; i++) {
        seedRN[i] = 0;
    }
    for (int i = 0; i < 600; i++) {
        spikeTimeRN[i] = -10;
    }
    for (int i = 0; i < 600; i++) {
        seedRN[i] = rand();
    }

    // synapse variables
    for (int i = 0; i < 180; i++) {
        inSynANAN[i] = 0.000000f;
    }
    for (int i = 0; i < 32400; i++) {
        gANAN[i] = 0;
    }
    for (int i = 0; i < 180; i++) {
        inSynPNAN[i] = 0.000000f;
    }
    for (int i = 0; i < 108000; i++) {
        gPNAN[i] = 0;
    }
    for (int i = 0; i < 600; i++) {
        inSynPNPN[i] = 0.000000f;
    }
    for (int i = 0; i < 360000; i++) {
        gPNPN[i] = 0;
    }
    for (int i = 0; i < 600; i++) {
        inSynRNPN[i] = 0.000000f;
    }
    for (int i = 0; i < 360000; i++) {
        gRNPN[i] = 0;
    }


    copyStateToDevice();

    //initializeAllSparseArrays(); //I comment this out instead of removing to keep in mind that sparse arrays need to be initialised manually by hand later
}

void initializeAllSparseArrays() {
}

void initSchmuker_2014_classifier()
 {
    
}

    void freeMem()
{
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntAN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkAN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntEvntAN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvntAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkEvntAN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvntAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(VAN));
    CHECK_CUDA_ERRORS(cudaFree(d_VAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(preVAN));
    CHECK_CUDA_ERRORS(cudaFree(d_preVAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntPN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkPN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntEvntPN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvntPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkEvntPN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvntPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(VPN));
    CHECK_CUDA_ERRORS(cudaFree(d_VPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(preVPN));
    CHECK_CUDA_ERRORS(cudaFree(d_preVPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntRN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntRN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkRN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkRN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntEvntRN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvntRN));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkEvntRN));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvntRN));
    CHECK_CUDA_ERRORS(cudaFreeHost(VRN));
    CHECK_CUDA_ERRORS(cudaFree(d_VRN));
    CHECK_CUDA_ERRORS(cudaFreeHost(seedRN));
    CHECK_CUDA_ERRORS(cudaFree(d_seedRN));
    CHECK_CUDA_ERRORS(cudaFreeHost(spikeTimeRN));
    CHECK_CUDA_ERRORS(cudaFree(d_spikeTimeRN));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynANAN));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynANAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(gANAN));
    CHECK_CUDA_ERRORS(cudaFree(d_gANAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynPNAN));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynPNAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(gPNAN));
    CHECK_CUDA_ERRORS(cudaFree(d_gPNAN));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynPNPN));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynPNPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(gPNPN));
    CHECK_CUDA_ERRORS(cudaFree(d_gPNPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynRNPN));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynRNPN));
    CHECK_CUDA_ERRORS(cudaFreeHost(gRNPN));
    CHECK_CUDA_ERRORS(cudaFree(d_gRNPN));
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
