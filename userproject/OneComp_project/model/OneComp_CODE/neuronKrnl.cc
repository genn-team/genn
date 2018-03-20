

#ifndef _OneComp_neuronKrnl_cc
#define _OneComp_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model OneComp containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(float t)
 {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shSpk[32];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (id == 0) {
        dd_glbSpkCntIzh1[0] = 0;
    }
    __threadfence();
    
    if (threadIdx.x == 0) {
        spkCount = 0;
    }
    __syncthreads();
    
    // neuron group Izh1
    if (id < 32) {
        
        // only do this for existing neurons
        if (id < 1) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VIzh1[id];
            scalar lU = dd_UIzh1[id];
            
            float Isyn = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f) {
                lV=(-6.50000000000000000e+01f);
                lU+=(6.00000000000000000e+00f);
            }
            lV += 0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+(4.00000000000000000e+00f)+Isyn)*DT; //at two times for numerical stability
            lV += 0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+(4.00000000000000000e+00f)+Isyn)*DT;
            lU += (2.00000000000000004e-02f)*((2.00000000000000011e-01f)*lV-lU)*DT;
            //if (lV > 30.0f) { // keep this only for visualisation -- not really necessaary otherwise
            //    lV = 30.0f;
            //}
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
            }
            dd_VIzh1[id] = lV;
            dd_UIzh1[id] = lU;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntIzh1[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkIzh1[posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
}

#endif
