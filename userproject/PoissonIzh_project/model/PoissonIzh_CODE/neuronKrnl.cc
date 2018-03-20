

#ifndef _PoissonIzh_neuronKrnl_cc
#define _PoissonIzh_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model PoissonIzh containing the neuron kernel function.
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
    
    if (threadIdx.x == 0) {
        spkCount = 0;
    }
    __syncthreads();
    
    // neuron group Izh1
    if (id < 32) {
        
        // only do this for existing neurons
        if (id < 10) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VIzh1[id];
            scalar lU = dd_UIzh1[id];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynPNIzh1 = dd_inSynPNIzh1[id];
            Isyn += linSynPNIzh1; linSynPNIzh1 = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=(-6.50000000000000000e+01f);
               lU+=(6.00000000000000000e+00f);
            } 
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+Isyn)*DT;
            lU+=(2.00000000000000004e-02f)*((2.00000000000000011e-01f)*lV-lU)*DT;
            //if (lV > 30.0f){   //keep this only for visualisation -- not really necessaary otherwise 
            //  lV=30.0f; 
            //}
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
            }
            dd_VIzh1[id] = lV;
            dd_UIzh1[id] = lU;
            // the post-synaptic dynamics
            
            dd_inSynPNIzh1[id] = linSynPNIzh1;
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
    
    // neuron group PN
    if ((id >= 32) && (id < 160)) {
        unsigned int lid = id - 32;
        
        // only do this for existing neurons
        if (lid < 100) {
            // pull neuron variables in a coalesced access
            scalar ltimeStepToSpike = dd_timeStepToSpikePN[lid];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= (ltimeStepToSpike <= 0.0f);
            // calculate membrane potential
            if(ltimeStepToSpike <= 0.0f) {
                ltimeStepToSpike += (5.00000000000000000e+01f) * exponentialDistDouble(&dd_rngPN[lid]);
            }
            ltimeStepToSpike -= 1.0f;
            
            // test for and register a true spike
            if ((ltimeStepToSpike <= 0.0f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
            }
            dd_timeStepToSpikePN[lid] = ltimeStepToSpike;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntPN[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkPN[posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
}

#endif
