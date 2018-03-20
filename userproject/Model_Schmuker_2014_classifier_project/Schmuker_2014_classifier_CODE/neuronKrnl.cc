

#ifndef _Schmuker_2014_classifier_neuronKrnl_cc
#define _Schmuker_2014_classifier_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model Schmuker_2014_classifier containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(unsigned int offsetRN, uint64_t * ratesRN, float t)
 {
    unsigned int id = 64 * blockIdx.x + threadIdx.x;
    __shared__ volatile unsigned int posSpkEvnt;
    __shared__ unsigned int shSpkEvnt[64];
    unsigned int spkEvntIdx;
    __shared__ volatile unsigned int spkEvntCount;
    __shared__ unsigned int shSpk[64];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (threadIdx.x == 0) {
        spkCount = 0;
        }
    if (threadIdx.x == 1) {
        spkEvntCount = 0;
        }
    __syncthreads();
    
    // neuron group AN
    if (id < 192) {
        
        // only do this for existing neurons
        if (id < 180) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VAN[id];
            scalar lpreV = dd_preVAN[id];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynPNAN = dd_inSynPNAN[id];
            Isyn += linSynPNAN * ((0.00000000000000000e+00f) - lV);
            // pull inSyn values in a coalesced access
            float linSynANAN = dd_inSynANAN[id];
            Isyn += linSynANAN * ((-9.20000000000000000e+01f) - lV);
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (3.19200000000000159e+01f));
            // calculate membrane potential
            if (lV <= 0) {
   lpreV= lV;
   lV= (1.08000000000000000e+04f)/(((6.00000000000000000e+01f)) - lV - ((1.65000000000000008e-02f))*Isyn) +((-1.48079999999999984e+02f));
}
else {   if ((lV < (3.19200000000000159e+01f)) && (lpreV <= 0)) {
       lpreV= lV;
       lV= (3.19200000000000159e+01f);
   }
   else {
       lpreV= lV;
       lV= -((6.00000000000000000e+01f));
   }
}

            bool spikeLikeEvent = false;
             {
                spikeLikeEvent |= (lV > (-3.50000000000000000e+01f));
                }
            // register a spike-like event
            if (spikeLikeEvent) {
                spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);
                shSpkEvnt[spkEvntIdx] = id;
                }
            // test for and register a true spike
            if ((lV >= (3.19200000000000159e+01f)) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
                }
            dd_VAN[id] = lV;
            dd_preVAN[id] = lpreV;
            // the post-synaptic dynamics
            linSynPNAN*=(6.06530659712633424e-01f);
            dd_inSynPNAN[id] = linSynPNAN;
            linSynANAN*=(9.39413062813475808e-01f);
            dd_inSynANAN[id] = linSynANAN;
            }
        __syncthreads();
        if (threadIdx.x == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntAN[0], spkEvntCount);
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntAN[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkEvntCount) {
            dd_glbSpkEvntAN[posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];
            }
        if (threadIdx.x < spkCount) {
            dd_glbSpkAN[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            }
        }
    
    // neuron group PN
    if ((id >= 192) && (id < 832)) {
        unsigned int lid = id - 192;
        
        // only do this for existing neurons
        if (lid < 600) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VPN[lid];
            scalar lpreV = dd_preVPN[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynRNPN = dd_inSynRNPN[lid];
            Isyn += linSynRNPN * ((0.00000000000000000e+00f) - lV);
            // pull inSyn values in a coalesced access
            float linSynPNPN = dd_inSynPNPN[lid];
            Isyn += linSynPNPN * ((-9.20000000000000000e+01f) - lV);
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (3.19200000000000159e+01f));
            // calculate membrane potential
            if (lV <= 0) {
   lpreV= lV;
   lV= (1.08000000000000000e+04f)/(((6.00000000000000000e+01f)) - lV - ((1.65000000000000008e-02f))*Isyn) +((-1.48079999999999984e+02f));
}
else {   if ((lV < (3.19200000000000159e+01f)) && (lpreV <= 0)) {
       lpreV= lV;
       lV= (3.19200000000000159e+01f);
   }
   else {
       lpreV= lV;
       lV= -((6.00000000000000000e+01f));
   }
}

            bool spikeLikeEvent = false;
             {
                spikeLikeEvent |= (lV > (-2.00000000000000000e+01f));
                }
             {
                spikeLikeEvent |= (lV > (-3.50000000000000000e+01f));
                }
            // register a spike-like event
            if (spikeLikeEvent) {
                spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);
                shSpkEvnt[spkEvntIdx] = lid;
                }
            // test for and register a true spike
            if ((lV >= (3.19200000000000159e+01f)) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
                }
            dd_VPN[lid] = lV;
            dd_preVPN[lid] = lpreV;
            // the post-synaptic dynamics
            linSynRNPN*=(6.06530659712633424e-01f);
            dd_inSynRNPN[lid] = linSynRNPN;
            linSynPNPN*=(9.13100716282262304e-01f);
            dd_inSynPNPN[lid] = linSynPNPN;
            }
        __syncthreads();
        if (threadIdx.x == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntPN[0], spkEvntCount);
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntPN[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkEvntCount) {
            dd_glbSpkEvntPN[posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];
            }
        if (threadIdx.x < spkCount) {
            dd_glbSpkPN[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            }
        }
    
    // neuron group RN
    if ((id >= 832) && (id < 1472)) {
        unsigned int lid = id - 832;
        
        // only do this for existing neurons
        if (lid < 600) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VRN[lid];
            uint64_t lseed = dd_seedRN[lid];
            scalar lspikeTime = dd_spikeTimeRN[lid];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (2.00000000000000000e+01f));
            // calculate membrane potential
            uint64_t theRnd;
if (lV > (-6.00000000000000000e+01f)) {
   lV= (-6.00000000000000000e+01f);
}else if (t - lspikeTime > ((2.50000000000000000e+00f))) {
   MYRAND(lseed,theRnd);
   if (theRnd < *(ratesRN+offsetRN+lid)) {
       lV= (2.00000000000000000e+01f);
       lspikeTime= t;
   }
}

            bool spikeLikeEvent = false;
             {
                spikeLikeEvent |= (lV > (-2.00000000000000000e+01f));
                }
            // register a spike-like event
            if (spikeLikeEvent) {
                spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);
                shSpkEvnt[spkEvntIdx] = lid;
                }
            // test for and register a true spike
            if ((lV >= (2.00000000000000000e+01f)) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
                }
            dd_VRN[lid] = lV;
            dd_seedRN[lid] = lseed;
            dd_spikeTimeRN[lid] = lspikeTime;
            }
        __syncthreads();
        if (threadIdx.x == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntRN[0], spkEvntCount);
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntRN[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkEvntCount) {
            dd_glbSpkEvntRN[posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];
            }
        if (threadIdx.x < spkCount) {
            dd_glbSpkRN[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            }
        }
    
    }

    #endif
