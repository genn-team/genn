

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model Izh_sparse containing general control code.
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

std::mt19937 rng;
std::uniform_real_distribution<float> standardUniformDistribution(0.000000f, 1.000000f);
std::normal_distribution<float> standardNormalDistribution(0.000000f, 1.000000f);
std::exponential_distribution<float> standardExponentialDistribution(1.000000f);
curandStatePhilox4_32_10_t *d_rng;
__device__ curandStatePhilox4_32_10_t *dd_rng;

// ------------------------------------------------------------------------
// neuron variables

__device__ volatile unsigned int d_done;
unsigned int * glbSpkCntPExc;
unsigned int * d_glbSpkCntPExc;
__device__ unsigned int * dd_glbSpkCntPExc;
unsigned int * glbSpkPExc;
unsigned int * d_glbSpkPExc;
__device__ unsigned int * dd_glbSpkPExc;
curandState *d_rngPExc;
__device__ curandState *dd_rngPExc;
scalar * VPExc;
scalar * d_VPExc;
__device__ scalar * dd_VPExc;
scalar * UPExc;
scalar * d_UPExc;
__device__ scalar * dd_UPExc;
scalar * aPExc;
scalar * d_aPExc;
__device__ scalar * dd_aPExc;
scalar * bPExc;
scalar * d_bPExc;
__device__ scalar * dd_bPExc;
scalar * cPExc;
scalar * d_cPExc;
__device__ scalar * dd_cPExc;
scalar * dPExc;
scalar * d_dPExc;
__device__ scalar * dd_dPExc;
unsigned int * glbSpkCntPInh;
unsigned int * d_glbSpkCntPInh;
__device__ unsigned int * dd_glbSpkCntPInh;
unsigned int * glbSpkPInh;
unsigned int * d_glbSpkPInh;
__device__ unsigned int * dd_glbSpkPInh;
curandState *d_rngPInh;
__device__ curandState *dd_rngPInh;
scalar * VPInh;
scalar * d_VPInh;
__device__ scalar * dd_VPInh;
scalar * UPInh;
scalar * d_UPInh;
__device__ scalar * dd_UPInh;
scalar * aPInh;
scalar * d_aPInh;
__device__ scalar * dd_aPInh;
scalar * bPInh;
scalar * d_bPInh;
__device__ scalar * dd_bPInh;
scalar * cPInh;
scalar * d_cPInh;
__device__ scalar * dd_cPInh;
scalar * dPInh;
scalar * d_dPInh;
__device__ scalar * dd_dPInh;

// ------------------------------------------------------------------------
// synapse variables

float * inSynExc_Exc;
float * d_inSynExc_Exc;
__device__ float * dd_inSynExc_Exc;
SparseProjection CExc_Exc;
unsigned int *d_indInGExc_Exc;
__device__ unsigned int *dd_indInGExc_Exc;
unsigned int *d_indExc_Exc;
__device__ unsigned int *dd_indExc_Exc;
scalar * gExc_Exc;
scalar * d_gExc_Exc;
__device__ scalar * dd_gExc_Exc;
float * inSynExc_Inh;
float * d_inSynExc_Inh;
__device__ float * dd_inSynExc_Inh;
SparseProjection CExc_Inh;
unsigned int *d_indInGExc_Inh;
__device__ unsigned int *dd_indInGExc_Inh;
unsigned int *d_indExc_Inh;
__device__ unsigned int *dd_indExc_Inh;
scalar * gExc_Inh;
scalar * d_gExc_Inh;
__device__ scalar * dd_gExc_Inh;
float * inSynInh_Exc;
float * d_inSynInh_Exc;
__device__ float * dd_inSynInh_Exc;
SparseProjection CInh_Exc;
unsigned int *d_indInGInh_Exc;
__device__ unsigned int *dd_indInGInh_Exc;
unsigned int *d_indInh_Exc;
__device__ unsigned int *dd_indInh_Exc;
scalar * gInh_Exc;
scalar * d_gInh_Exc;
__device__ scalar * dd_gInh_Exc;
float * inSynInh_Inh;
float * d_inSynInh_Inh;
__device__ float * dd_inSynInh_Inh;
SparseProjection CInh_Inh;
unsigned int *d_indInGInh_Inh;
__device__ unsigned int *dd_indInGInh_Inh;
unsigned int *d_indInh_Inh;
__device__ unsigned int *dd_indInh_Inh;
scalar * gInh_Inh;
scalar * d_gInh_Inh;
__device__ scalar * dd_gInh_Inh;

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
    deviceMemAllocate(&d_rng, dd_rng, 1 * sizeof(curandStatePhilox4_32_10_t));
    cudaHostAlloc(&glbSpkCntPExc, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntPExc, dd_glbSpkCntPExc, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkPExc, 8000 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkPExc, dd_glbSpkPExc, 8000 * sizeof(unsigned int));
    deviceMemAllocate(&d_rngPExc, dd_rngPExc, 8000 * sizeof(curandState));
    cudaHostAlloc(&VPExc, 8000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VPExc, dd_VPExc, 8000 * sizeof(scalar));
    cudaHostAlloc(&UPExc, 8000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_UPExc, dd_UPExc, 8000 * sizeof(scalar));
    cudaHostAlloc(&aPExc, 8000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_aPExc, dd_aPExc, 8000 * sizeof(scalar));
    cudaHostAlloc(&bPExc, 8000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_bPExc, dd_bPExc, 8000 * sizeof(scalar));
    cudaHostAlloc(&cPExc, 8000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_cPExc, dd_cPExc, 8000 * sizeof(scalar));
    cudaHostAlloc(&dPExc, 8000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_dPExc, dd_dPExc, 8000 * sizeof(scalar));

    cudaHostAlloc(&glbSpkCntPInh, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntPInh, dd_glbSpkCntPInh, 1 * sizeof(unsigned int));
    cudaHostAlloc(&glbSpkPInh, 2000 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkPInh, dd_glbSpkPInh, 2000 * sizeof(unsigned int));
    deviceMemAllocate(&d_rngPInh, dd_rngPInh, 2000 * sizeof(curandState));
    cudaHostAlloc(&VPInh, 2000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_VPInh, dd_VPInh, 2000 * sizeof(scalar));
    cudaHostAlloc(&UPInh, 2000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_UPInh, dd_UPInh, 2000 * sizeof(scalar));
    cudaHostAlloc(&aPInh, 2000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_aPInh, dd_aPInh, 2000 * sizeof(scalar));
    cudaHostAlloc(&bPInh, 2000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_bPInh, dd_bPInh, 2000 * sizeof(scalar));
    cudaHostAlloc(&cPInh, 2000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_cPInh, dd_cPInh, 2000 * sizeof(scalar));
    cudaHostAlloc(&dPInh, 2000 * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_dPInh, dd_dPInh, 2000 * sizeof(scalar));

    cudaHostAlloc(&inSynExc_Exc, 8000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynExc_Exc, dd_inSynExc_Exc, 8000 * sizeof(float));

    cudaHostAlloc(&inSynExc_Inh, 2000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynExc_Inh, dd_inSynExc_Inh, 2000 * sizeof(float));

    cudaHostAlloc(&inSynInh_Exc, 8000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynInh_Exc, dd_inSynInh_Exc, 8000 * sizeof(float));

    cudaHostAlloc(&inSynInh_Inh, 2000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynInh_Inh, dd_inSynInh_Inh, 2000 * sizeof(float));

}

void allocateExc_Exc(unsigned int connN){
// Allocate host side variables
  CExc_Exc.connN= connN;
    cudaHostAlloc(&CExc_Exc.indInG, 8001 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CExc_Exc.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CExc_Exc.preInd= NULL;
  CExc_Exc.revIndInG= NULL;
  CExc_Exc.revInd= NULL;
  CExc_Exc.remap= NULL;
    deviceMemAllocate(&d_indInGExc_Exc, dd_indInGExc_Exc, 8001 * sizeof(unsigned int));
    deviceMemAllocate(&d_indExc_Exc, dd_indExc_Exc, CExc_Exc.connN * sizeof(unsigned int));
    cudaHostAlloc(&gExc_Exc, CExc_Exc.connN * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gExc_Exc, dd_gExc_Exc, CExc_Exc.connN * sizeof(scalar));
}

void createSparseConnectivityFromDenseExc_Exc(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseExc_Exc() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateExc_Exc(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateExc_Inh(unsigned int connN){
// Allocate host side variables
  CExc_Inh.connN= connN;
    cudaHostAlloc(&CExc_Inh.indInG, 8001 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CExc_Inh.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CExc_Inh.preInd= NULL;
  CExc_Inh.revIndInG= NULL;
  CExc_Inh.revInd= NULL;
  CExc_Inh.remap= NULL;
    deviceMemAllocate(&d_indInGExc_Inh, dd_indInGExc_Inh, 8001 * sizeof(unsigned int));
    deviceMemAllocate(&d_indExc_Inh, dd_indExc_Inh, CExc_Inh.connN * sizeof(unsigned int));
    cudaHostAlloc(&gExc_Inh, CExc_Inh.connN * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gExc_Inh, dd_gExc_Inh, CExc_Inh.connN * sizeof(scalar));
}

void createSparseConnectivityFromDenseExc_Inh(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseExc_Inh() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateExc_Inh(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateInh_Exc(unsigned int connN){
// Allocate host side variables
  CInh_Exc.connN= connN;
    cudaHostAlloc(&CInh_Exc.indInG, 2001 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CInh_Exc.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CInh_Exc.preInd= NULL;
  CInh_Exc.revIndInG= NULL;
  CInh_Exc.revInd= NULL;
  CInh_Exc.remap= NULL;
    deviceMemAllocate(&d_indInGInh_Exc, dd_indInGInh_Exc, 2001 * sizeof(unsigned int));
    deviceMemAllocate(&d_indInh_Exc, dd_indInh_Exc, CInh_Exc.connN * sizeof(unsigned int));
    cudaHostAlloc(&gInh_Exc, CInh_Exc.connN * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gInh_Exc, dd_gInh_Exc, CInh_Exc.connN * sizeof(scalar));
}

void createSparseConnectivityFromDenseInh_Exc(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseInh_Exc() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateInh_Exc(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateInh_Inh(unsigned int connN){
// Allocate host side variables
  CInh_Inh.connN= connN;
    cudaHostAlloc(&CInh_Inh.indInG, 2001 * sizeof(unsigned int), cudaHostAllocPortable);
    cudaHostAlloc(&CInh_Inh.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CInh_Inh.preInd= NULL;
  CInh_Inh.revIndInG= NULL;
  CInh_Inh.revInd= NULL;
  CInh_Inh.remap= NULL;
    deviceMemAllocate(&d_indInGInh_Inh, dd_indInGInh_Inh, 2001 * sizeof(unsigned int));
    deviceMemAllocate(&d_indInh_Inh, dd_indInh_Inh, CInh_Inh.connN * sizeof(unsigned int));
    cudaHostAlloc(&gInh_Inh, CInh_Inh.connN * sizeof(scalar), cudaHostAllocPortable);
    deviceMemAllocate(&d_gInh_Inh, dd_gInh_Inh, CInh_Inh.connN * sizeof(scalar));
}

void createSparseConnectivityFromDenseInh_Inh(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseInh_Inh() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateInh_Inh(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void freeMem()
{
    CHECK_CUDA_ERRORS(cudaFree(d_rng));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntPExc));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_rngPExc));
    CHECK_CUDA_ERRORS(cudaFreeHost(VPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_VPExc));
    CHECK_CUDA_ERRORS(cudaFreeHost(UPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_UPExc));
    CHECK_CUDA_ERRORS(cudaFreeHost(aPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_aPExc));
    CHECK_CUDA_ERRORS(cudaFreeHost(bPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_bPExc));
    CHECK_CUDA_ERRORS(cudaFreeHost(cPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_cPExc));
    CHECK_CUDA_ERRORS(cudaFreeHost(dPExc));
    CHECK_CUDA_ERRORS(cudaFree(d_dPExc));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntPInh));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_rngPInh));
    CHECK_CUDA_ERRORS(cudaFreeHost(VPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_VPInh));
    CHECK_CUDA_ERRORS(cudaFreeHost(UPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_UPInh));
    CHECK_CUDA_ERRORS(cudaFreeHost(aPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_aPInh));
    CHECK_CUDA_ERRORS(cudaFreeHost(bPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_bPInh));
    CHECK_CUDA_ERRORS(cudaFreeHost(cPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_cPInh));
    CHECK_CUDA_ERRORS(cudaFreeHost(dPInh));
    CHECK_CUDA_ERRORS(cudaFree(d_dPInh));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynExc_Exc));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynExc_Exc));
    CExc_Exc.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CExc_Exc.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGExc_Exc));
    CHECK_CUDA_ERRORS(cudaFreeHost(CExc_Exc.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indExc_Exc));
    CHECK_CUDA_ERRORS(cudaFreeHost(gExc_Exc));
    CHECK_CUDA_ERRORS(cudaFree(d_gExc_Exc));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynExc_Inh));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynExc_Inh));
    CExc_Inh.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CExc_Inh.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGExc_Inh));
    CHECK_CUDA_ERRORS(cudaFreeHost(CExc_Inh.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indExc_Inh));
    CHECK_CUDA_ERRORS(cudaFreeHost(gExc_Inh));
    CHECK_CUDA_ERRORS(cudaFree(d_gExc_Inh));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynInh_Exc));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynInh_Exc));
    CInh_Exc.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CInh_Exc.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGInh_Exc));
    CHECK_CUDA_ERRORS(cudaFreeHost(CInh_Exc.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indInh_Exc));
    CHECK_CUDA_ERRORS(cudaFreeHost(gInh_Exc));
    CHECK_CUDA_ERRORS(cudaFree(d_gInh_Exc));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynInh_Inh));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynInh_Inh));
    CInh_Inh.connN= 0;
    CHECK_CUDA_ERRORS(cudaFreeHost(CInh_Inh.indInG));
    CHECK_CUDA_ERRORS(cudaFree(d_indInGInh_Inh));
    CHECK_CUDA_ERRORS(cudaFreeHost(CInh_Inh.ind));
    CHECK_CUDA_ERRORS(cudaFree(d_indInh_Inh));
    CHECK_CUDA_ERRORS(cudaFreeHost(gInh_Inh));
    CHECK_CUDA_ERRORS(cudaFree(d_gInh_Inh));
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
