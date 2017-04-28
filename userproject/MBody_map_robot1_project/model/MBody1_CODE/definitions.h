

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model MBody1 containing useful Macros used for both GPU amd CPU versions.
*/
//-------------------------------------------------------------------------

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include "utils.h"
#include "sparseUtils.h"

#include "sparseProjection.h"
#include <stdint.h>

#define __device__
#define __global__
#define __host__
#define __constant__
#define __shared__
#undef DT
#define DT 0.500000f
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

// ------------------------------------------------------------------------
// global variables

extern unsigned long long iT;
extern float t;

// ------------------------------------------------------------------------
// neuron variables

extern unsigned int * glbSpkCntDN;
extern unsigned int * glbSpkDN;
extern unsigned int * glbSpkCntEvntDN;
extern unsigned int * glbSpkEvntDN;
extern float * sTDN;
extern scalar * VDN;
extern scalar * preVDN;
extern unsigned int * glbSpkCntKC;
extern unsigned int * glbSpkKC;
extern float * sTKC;
extern scalar * VKC;
extern scalar * preVKC;
extern unsigned int * glbSpkCntLHI;
extern unsigned int * glbSpkLHI;
extern unsigned int * glbSpkCntEvntLHI;
extern unsigned int * glbSpkEvntLHI;
extern scalar * VLHI;
extern scalar * preVLHI;
extern unsigned int * glbSpkCntPN;
extern unsigned int * glbSpkPN;
extern scalar * VPN;
extern uint64_t * seedPN;
extern scalar * spikeTimePN;
extern uint64_t * ratesPN;
extern unsigned int offsetPN;

#define glbSpkShiftDN 0
#define glbSpkShiftKC 0
#define glbSpkShiftLHI 0
#define glbSpkShiftPN 0
#define spikeCount_DN glbSpkCntDN[0]
#define spike_DN glbSpkDN
#define spikeEventCount_DN glbSpkCntEvntDN[0]
#define spikeEvent_DN glbSpkEvntDN
#define spikeCount_KC glbSpkCntKC[0]
#define spike_KC glbSpkKC
#define spikeCount_LHI glbSpkCntLHI[0]
#define spike_LHI glbSpkLHI
#define spikeEventCount_LHI glbSpkCntEvntLHI[0]
#define spikeEvent_LHI glbSpkEvntLHI
#define spikeCount_PN glbSpkCntPN[0]
#define spike_PN glbSpkPN

// ------------------------------------------------------------------------
// synapse variables

extern float * inSynDNDN;
extern float * inSynKCDN;
extern scalar * gKCDN;
extern scalar * gRawKCDN;
extern float * inSynLHIKC;
extern float * inSynPNKC;
extern scalar * gPNKC;
extern float * inSynPNLHI;
extern scalar * gPNLHI;

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/

// ------------------------------------------------------------------------
// Function for setting the CUDA device and the host's global variables.
// Also estimates memory usage on device.

void allocateMem();

// ------------------------------------------------------------------------
// Function to (re)set all model variables to their compile-time, homogeneous initial
// values. Note that this typically includes synaptic weight values. The function
// (re)sets host side variables and copies them to the GPU device.

void initialize();

// ------------------------------------------------------------------------
// initialization of variables, e.g. reverse sparse arrays etc.
// that the user would not want to worry about

void initMBody1();

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
// Throw an error for "old style" time stepping calls (using CPU)

template <class T>
void stepTimeCPU(T arg1, ...) {
    gennError("Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.");
    }

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)

void stepTimeCPU();

#endif
