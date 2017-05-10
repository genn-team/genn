

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

// ------------------------------------------------------------------------
// global variables

unsigned long long iT= 0;
float t;

// ------------------------------------------------------------------------
// neuron variables

unsigned int * glbSpkCntDN;
unsigned int * glbSpkDN;
unsigned int * glbSpkCntEvntDN;
unsigned int * glbSpkEvntDN;
float * sTDN;
scalar * VDN;
scalar * preVDN;
unsigned int * glbSpkCntKC;
unsigned int * glbSpkKC;
float * sTKC;
scalar * VKC;
scalar * preVKC;
unsigned int * glbSpkCntLHI;
unsigned int * glbSpkLHI;
unsigned int * glbSpkCntEvntLHI;
unsigned int * glbSpkEvntLHI;
scalar * VLHI;
scalar * preVLHI;
unsigned int * glbSpkCntPN;
unsigned int * glbSpkPN;
scalar * VPN;
uint64_t * seedPN;
scalar * spikeTimePN;
uint64_t * ratesPN;
unsigned int offsetPN;

// ------------------------------------------------------------------------
// synapse variables

float * inSynDNDN;
float * inSynKCDN;
scalar * gKCDN;
scalar * gRawKCDN;
float * inSynLHIKC;
float * inSynPNKC;
scalar * gPNKC;
float * inSynPNLHI;
scalar * gPNLHI;

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

#include "neuronFnct.cc"
#include "synapseFnct.cc"
void allocateMem()
{
    glbSpkCntDN = new unsigned int[1];
    glbSpkDN = new unsigned int[10];
    glbSpkCntEvntDN = new unsigned int[1];
    glbSpkEvntDN = new unsigned int[10];
    sTDN = new float[10];
    VDN = new scalar[10];
    preVDN = new scalar[10];

    glbSpkCntKC = new unsigned int[1];
    glbSpkKC = new unsigned int[5000];
    sTKC = new float[5000];
    VKC = new scalar[5000];
    preVKC = new scalar[5000];

    glbSpkCntLHI = new unsigned int[1];
    glbSpkLHI = new unsigned int[20];
    glbSpkCntEvntLHI = new unsigned int[1];
    glbSpkEvntLHI = new unsigned int[20];
    VLHI = new scalar[20];
    preVLHI = new scalar[20];

    glbSpkCntPN = new unsigned int[1];
    glbSpkPN = new unsigned int[1024];
    VPN = new scalar[1024];
    seedPN = new uint64_t[1024];
    spikeTimePN = new scalar[1024];

    inSynDNDN = new float[10];

    inSynKCDN = new float[10];
    gKCDN = new scalar[50000];
    gRawKCDN = new scalar[50000];

    inSynLHIKC = new float[5000];

    inSynPNKC = new float[5000];
    gPNKC = new scalar[5120000];

    inSynPNLHI = new float[20];
    gPNLHI = new scalar[20480];

}

//-------------------------------------------------------------------------
/*! \brief Function to (re)set all model variables to their compile-time, homogeneous initial values.
 Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device.
*/
//-------------------------------------------------------------------------

void initialize()
{
    srand((unsigned int) 1234);

    // neuron variables
    glbSpkCntDN[0] = 0;
    for (int i = 0; i < 10; i++) {
        glbSpkDN[i] = 0;
    }
    glbSpkCntEvntDN[0] = 0;
    for (int i = 0; i < 10; i++) {
        glbSpkEvntDN[i] = 0;
    }
    for (int i = 0; i < 10; i++) {
        sTDN[i] = -10.0;
    }
    for (int i = 0; i < 10; i++) {
        VDN[i] = -60;
    }
    for (int i = 0; i < 10; i++) {
        preVDN[i] = -60;
    }
    glbSpkCntKC[0] = 0;
    for (int i = 0; i < 5000; i++) {
        glbSpkKC[i] = 0;
    }
    for (int i = 0; i < 5000; i++) {
        sTKC[i] = -10.0;
    }
    for (int i = 0; i < 5000; i++) {
        VKC[i] = -60;
    }
    for (int i = 0; i < 5000; i++) {
        preVKC[i] = -60;
    }
    glbSpkCntLHI[0] = 0;
    for (int i = 0; i < 20; i++) {
        glbSpkLHI[i] = 0;
    }
    glbSpkCntEvntLHI[0] = 0;
    for (int i = 0; i < 20; i++) {
        glbSpkEvntLHI[i] = 0;
    }
    for (int i = 0; i < 20; i++) {
        VLHI[i] = -60;
    }
    for (int i = 0; i < 20; i++) {
        preVLHI[i] = -60;
    }
    glbSpkCntPN[0] = 0;
    for (int i = 0; i < 1024; i++) {
        glbSpkPN[i] = 0;
    }
    for (int i = 0; i < 1024; i++) {
        VPN[i] = -60;
    }
    for (int i = 0; i < 1024; i++) {
        seedPN[i] = 0;
    }
    for (int i = 0; i < 1024; i++) {
        spikeTimePN[i] = -10;
    }
    for (int i = 0; i < 1024; i++) {
        seedPN[i] = rand();
    }

    // synapse variables
    for (int i = 0; i < 10; i++) {
        inSynDNDN[i] = 0.000000f;
    }
    for (int i = 0; i < 10; i++) {
        inSynKCDN[i] = 0.000000f;
    }
    for (int i = 0; i < 50000; i++) {
        gKCDN[i] = 0.01;
    }
    for (int i = 0; i < 50000; i++) {
        gRawKCDN[i] = 0.01;
    }
    for (int i = 0; i < 5000; i++) {
        inSynLHIKC[i] = 0.000000f;
    }
    for (int i = 0; i < 5000; i++) {
        inSynPNKC[i] = 0.000000f;
    }
    for (int i = 0; i < 5120000; i++) {
        gPNKC[i] = 1;
    }
    for (int i = 0; i < 20; i++) {
        inSynPNLHI[i] = 0.000000f;
    }
    for (int i = 0; i < 20480; i++) {
        gPNLHI[i] = 0;
    }


}

void initMBody1()
 {
    
    }

void freeMem()
{
    delete[] glbSpkCntDN;
    delete[] glbSpkDN;
    delete[] glbSpkCntEvntDN;
    delete[] glbSpkEvntDN;
    delete[] sTDN;
    delete[] VDN;
    delete[] preVDN;
    delete[] glbSpkCntKC;
    delete[] glbSpkKC;
    delete[] sTKC;
    delete[] VKC;
    delete[] preVKC;
    delete[] glbSpkCntLHI;
    delete[] glbSpkLHI;
    delete[] glbSpkCntEvntLHI;
    delete[] glbSpkEvntLHI;
    delete[] VLHI;
    delete[] preVLHI;
    delete[] glbSpkCntPN;
    delete[] glbSpkPN;
    delete[] VPN;
    delete[] seedPN;
    delete[] spikeTimePN;
    delete[] inSynDNDN;
    delete[] inSynKCDN;
    delete[] gKCDN;
    delete[] gRawKCDN;
    delete[] inSynLHIKC;
    delete[] inSynPNKC;
    delete[] gPNKC;
    delete[] inSynPNLHI;
    delete[] gPNLHI;
}

void exitGeNN(){
  freeMem();
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
