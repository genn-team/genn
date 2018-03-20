

#ifndef _HHVClamp_neuronKrnl_cc
#define _HHVClamp_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model HHVClamp containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(scalar IsynGHH, scalar stepVGHH, float t)
 {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shSpk[32];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (id == 0) {
        dd_glbSpkCntHH[0] = 0;
    }
    __threadfence();
    
    if (threadIdx.x == 0) {
        spkCount = 0;
    }
    __syncthreads();
    
    // neuron group HH
    if (id < 32) {
        
        // only do this for existing neurons
        if (id < 12) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VHH[id];
            scalar lm = dd_mHH[id];
            scalar lh = dd_hHH[id];
            scalar ln = dd_nHH[id];
            scalar lgNa = dd_gNaHH[id];
            scalar lENa = dd_ENaHH[id];
            scalar lgK = dd_gKHH[id];
            scalar lEK = dd_EKHH[id];
            scalar lgl = dd_glHH[id];
            scalar lEl = dd_ElHH[id];
            scalar lC = dd_CHH[id];
            scalar lerr = dd_errHH[id];
            
            float Isyn = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV > 100);
            // calculate membrane potential
            scalar Imem;
            unsigned int mt;
            scalar mdt= DT/100.0f;
            scalar Icoupl;
            for (mt=0; mt < 100; mt++) {
               Icoupl= 200.0f*(stepVGHH-lV);
               Imem= -(lm*lm*lm*lh*lgNa*(lV-(lENa))+
                   ln*ln*ln*ln*lgK*(lV-(lEK))+
                   lgl*(lV-(lEl))-Icoupl);
               scalar _a= (3.5f+0.1f*lV) / (1.0f-expf(-3.5f-0.1f*lV));
               scalar _b= 4.0f*expf(-(lV+60.0f)/18.0f);
               lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
               _a= 0.07f*expf(-lV/20.0f-3.0f);
               _b= 1.0f / (expf(-3.0f-0.1f*lV)+1.0f);
               lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
               _a= (-0.5f-0.01f*lV) / (expf(-5.0f-0.1f*lV)-1.0f);
               _b= 0.125f*expf(-(lV+60.0f)/80.0f);
               ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
               lV+= Imem/lC*mdt;
            }
            lerr+= abs(Icoupl-IsynGHH);
            
            // test for and register a true spike
            if ((lV > 100) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
            }
            dd_VHH[id] = lV;
            dd_mHH[id] = lm;
            dd_hHH[id] = lh;
            dd_nHH[id] = ln;
            dd_gNaHH[id] = lgNa;
            dd_ENaHH[id] = lENa;
            dd_gKHH[id] = lgK;
            dd_EKHH[id] = lEK;
            dd_glHH[id] = lgl;
            dd_ElHH[id] = lEl;
            dd_CHH[id] = lC;
            dd_errHH[id] = lerr;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntHH[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkHH[posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
}

#endif
