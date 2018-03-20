

#ifndef _Izh_sparse_neuronKrnl_cc
#define _Izh_sparse_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model Izh_sparse containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(float t)
 {
    unsigned int id = 128 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shSpk[128];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (threadIdx.x == 0) {
        spkCount = 0;
    }
    __syncthreads();
    
    // neuron group PExc
    if (id < 8064) {
        
        // only do this for existing neurons
        if (id < 8000) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VPExc[id];
            scalar lU = dd_UPExc[id];
            scalar la = dd_aPExc[id];
            scalar lb = dd_bPExc[id];
            scalar lc = dd_cPExc[id];
            scalar ld = dd_dPExc[id];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynExc_Exc = dd_inSynExc_Exc[id];
            Isyn += linSynExc_Exc; linSynExc_Exc = 0;
            // pull inSyn values in a coalesced access
            float linSynInh_Exc = dd_inSynInh_Exc[id];
            Isyn += linSynInh_Exc; linSynInh_Exc = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=lc;
               lU+=ld;
            } 
            const scalar i0 = (0.00000000000000000e+00f) + (curand_normal(&dd_rngPExc[id]) * (5.00000000000000000e+00f));
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+i0+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+i0+Isyn)*DT;
            lU+=la*(lb*lV-lU)*DT;
            //if (lV > 30.0f){      //keep this only for visualisation -- not really necessaary otherwise 
            //  lV=30.0f; 
            //}
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
            }
            dd_VPExc[id] = lV;
            dd_UPExc[id] = lU;
            dd_aPExc[id] = la;
            dd_bPExc[id] = lb;
            dd_cPExc[id] = lc;
            dd_dPExc[id] = ld;
            // the post-synaptic dynamics
            
            dd_inSynExc_Exc[id] = linSynExc_Exc;
            
            dd_inSynInh_Exc[id] = linSynInh_Exc;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntPExc[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkPExc[posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
    // neuron group PInh
    if ((id >= 8064) && (id < 10112)) {
        unsigned int lid = id - 8064;
        
        // only do this for existing neurons
        if (lid < 2000) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VPInh[lid];
            scalar lU = dd_UPInh[lid];
            scalar la = dd_aPInh[lid];
            scalar lb = dd_bPInh[lid];
            scalar lc = dd_cPInh[lid];
            scalar ld = dd_dPInh[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynExc_Inh = dd_inSynExc_Inh[lid];
            Isyn += linSynExc_Inh; linSynExc_Inh = 0;
            // pull inSyn values in a coalesced access
            float linSynInh_Inh = dd_inSynInh_Inh[lid];
            Isyn += linSynInh_Inh; linSynInh_Inh = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=lc;
               lU+=ld;
            } 
            const scalar i0 = (0.00000000000000000e+00f) + (curand_normal(&dd_rngPInh[lid]) * (2.00000000000000000e+00f));
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+i0+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+i0+Isyn)*DT;
            lU+=la*(lb*lV-lU)*DT;
            //if (lV > 30.0f){      //keep this only for visualisation -- not really necessaary otherwise 
            //  lV=30.0f; 
            //}
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
            }
            dd_VPInh[lid] = lV;
            dd_UPInh[lid] = lU;
            dd_aPInh[lid] = la;
            dd_bPInh[lid] = lb;
            dd_cPInh[lid] = lc;
            dd_dPInh[lid] = ld;
            // the post-synaptic dynamics
            
            dd_inSynExc_Inh[lid] = linSynExc_Inh;
            
            dd_inSynInh_Inh[lid] = linSynInh_Inh;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntPInh[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkPInh[posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
}

#endif
