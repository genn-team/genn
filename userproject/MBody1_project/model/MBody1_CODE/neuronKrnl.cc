

#ifndef _MBody1_neuronKrnl_cc
#define _MBody1_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model MBody1 containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(unsigned int offsetPN, uint64_t * ratesPN, float t)
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
    
    // neuron group DN
    if (id < 128) {
        
        // only do this for existing neurons
        if (id < 100) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VDN[id];
            scalar lm = dd_mDN[id];
            scalar lh = dd_hDN[id];
            scalar ln = dd_nDN[id];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynKCDN = dd_inSynKCDN[id];
            Isyn += linSynKCDN * ((0.00000000000000000e+00f) - lV);
            // pull inSyn values in a coalesced access
            float linSynDNDN = dd_inSynDNDN[id];
            Isyn += linSynDNDN * ((-9.20000000000000000e+01f) - lV);
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 0.0f);
            // calculate membrane potential
            scalar Imem;
            unsigned int mt;
            scalar mdt= DT/25.0f;
            for (mt=0; mt < 25; mt++) {
               Imem= -(lm*lm*lm*lh*(7.15000000000000036e+00f)*(lV-((5.00000000000000000e+01f)))+
                   ln*ln*ln*ln*(1.42999999999999994e+00f)*(lV-((-9.50000000000000000e+01f)))+
                   (2.67200000000000007e-02f)*(lV-((-6.35630000000000024e+01f)))-Isyn);
               scalar _a;
               if (lV == -52.0f) {
                   _a= 1.28f;
               }
               else {
                   _a= 0.32f*(-52.0f-lV)/(expf((-52.0f-lV)/4.0f)-1.0f);
               }
               scalar _b;
               if (lV == -25.0f) {
                   _b= 1.4f;
               }
               else {
                   _b= 0.28f*(lV+25.0f)/(expf((lV+25.0f)/5.0f)-1.0f);
               }
               lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
               _a= 0.128f*expf((-48.0f-lV)/18.0f);
               _b= 4.0f / (expf((-25.0f-lV)/5.0f)+1.0f);
               lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
               if (lV == -50.0f) {
                   _a= 0.16f;
               }
               else {
                   _a= 0.032f*(-50.0f-lV)/(expf((-50.0f-lV)/5.0f)-1.0f);
               }
               _b= 0.5f*expf((-55.0f-lV)/40.0f);
               ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
               lV+= Imem/(1.42999999999999988e-01f)*mdt;
            }
            
            bool spikeLikeEvent = false;
             {
                spikeLikeEvent |= (lV > (-3.00000000000000000e+01f));
            }
            // register a spike-like event
            if (spikeLikeEvent) {
                spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);
                shSpkEvnt[spkEvntIdx] = id;
            }
            // test for and register a true spike
            if ((lV >= 0.0f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
            }
            dd_VDN[id] = lV;
            dd_mDN[id] = lm;
            dd_hDN[id] = lh;
            dd_nDN[id] = ln;
            // the post-synaptic dynamics
            linSynKCDN*=(9.80198673306755253e-01f);
            dd_inSynKCDN[id] = linSynKCDN;
            linSynDNDN*=(9.60789439152323177e-01f);
            dd_inSynDNDN[id] = linSynDNDN;
        }
        __syncthreads();
        if (threadIdx.x == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntDN[0], spkEvntCount);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntDN[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkEvntCount) {
            dd_glbSpkEvntDN[posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];
        }
        if (threadIdx.x < spkCount) {
            dd_glbSpkDN[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            dd_sTDN[shSpk[threadIdx.x]] = t;
        }
    }
    
    // neuron group KC
    if ((id >= 128) && (id < 1152)) {
        unsigned int lid = id - 128;
        
        // only do this for existing neurons
        if (lid < 1000) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VKC[lid];
            scalar lm = dd_mKC[lid];
            scalar lh = dd_hKC[lid];
            scalar ln = dd_nKC[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynPNKC = dd_inSynPNKC[lid];
            Isyn += linSynPNKC * ((0.00000000000000000e+00f) - lV);
            // pull inSyn values in a coalesced access
            float linSynLHIKC = dd_inSynLHIKC[lid];
            Isyn += linSynLHIKC * ((-9.20000000000000000e+01f) - lV);
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 0.0f);
            // calculate membrane potential
            scalar Imem;
            unsigned int mt;
            scalar mdt= DT/25.0f;
            for (mt=0; mt < 25; mt++) {
               Imem= -(lm*lm*lm*lh*(7.15000000000000036e+00f)*(lV-((5.00000000000000000e+01f)))+
                   ln*ln*ln*ln*(1.42999999999999994e+00f)*(lV-((-9.50000000000000000e+01f)))+
                   (2.67200000000000007e-02f)*(lV-((-6.35630000000000024e+01f)))-Isyn);
               scalar _a;
               if (lV == -52.0f) {
                   _a= 1.28f;
               }
               else {
                   _a= 0.32f*(-52.0f-lV)/(expf((-52.0f-lV)/4.0f)-1.0f);
               }
               scalar _b;
               if (lV == -25.0f) {
                   _b= 1.4f;
               }
               else {
                   _b= 0.28f*(lV+25.0f)/(expf((lV+25.0f)/5.0f)-1.0f);
               }
               lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
               _a= 0.128f*expf((-48.0f-lV)/18.0f);
               _b= 4.0f / (expf((-25.0f-lV)/5.0f)+1.0f);
               lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
               if (lV == -50.0f) {
                   _a= 0.16f;
               }
               else {
                   _a= 0.032f*(-50.0f-lV)/(expf((-50.0f-lV)/5.0f)-1.0f);
               }
               _b= 0.5f*expf((-55.0f-lV)/40.0f);
               ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
               lV+= Imem/(1.42999999999999988e-01f)*mdt;
            }
            
            // test for and register a true spike
            if ((lV >= 0.0f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
            }
            dd_VKC[lid] = lV;
            dd_mKC[lid] = lm;
            dd_hKC[lid] = lh;
            dd_nKC[lid] = ln;
            // the post-synaptic dynamics
            linSynPNKC*=(9.04837418035959518e-01f);
            dd_inSynPNKC[lid] = linSynPNKC;
            linSynLHIKC*=(9.35506985031617777e-01f);
            dd_inSynLHIKC[lid] = linSynLHIKC;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntKC[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkKC[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            dd_sTKC[shSpk[threadIdx.x]] = t;
        }
    }
    
    // neuron group LHI
    if ((id >= 1152) && (id < 1216)) {
        unsigned int lid = id - 1152;
        
        // only do this for existing neurons
        if (lid < 20) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VLHI[lid];
            scalar lm = dd_mLHI[lid];
            scalar lh = dd_hLHI[lid];
            scalar ln = dd_nLHI[lid];
            
            float Isyn = 0;
            // pull inSyn values in a coalesced access
            float linSynPNLHI = dd_inSynPNLHI[lid];
            Isyn += linSynPNLHI * ((0.00000000000000000e+00f) - lV);
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 0.0f);
            // calculate membrane potential
            scalar Imem;
            unsigned int mt;
            scalar mdt= DT/25.0f;
            for (mt=0; mt < 25; mt++) {
               Imem= -(lm*lm*lm*lh*(7.15000000000000036e+00f)*(lV-((5.00000000000000000e+01f)))+
                   ln*ln*ln*ln*(1.42999999999999994e+00f)*(lV-((-9.50000000000000000e+01f)))+
                   (2.67200000000000007e-02f)*(lV-((-6.35630000000000024e+01f)))-Isyn);
               scalar _a;
               if (lV == -52.0f) {
                   _a= 1.28f;
               }
               else {
                   _a= 0.32f*(-52.0f-lV)/(expf((-52.0f-lV)/4.0f)-1.0f);
               }
               scalar _b;
               if (lV == -25.0f) {
                   _b= 1.4f;
               }
               else {
                   _b= 0.28f*(lV+25.0f)/(expf((lV+25.0f)/5.0f)-1.0f);
               }
               lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
               _a= 0.128f*expf((-48.0f-lV)/18.0f);
               _b= 4.0f / (expf((-25.0f-lV)/5.0f)+1.0f);
               lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
               if (lV == -50.0f) {
                   _a= 0.16f;
               }
               else {
                   _a= 0.032f*(-50.0f-lV)/(expf((-50.0f-lV)/5.0f)-1.0f);
               }
               _b= 0.5f*expf((-55.0f-lV)/40.0f);
               ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
               lV+= Imem/(1.42999999999999988e-01f)*mdt;
            }
            
            bool spikeLikeEvent = false;
             {
                spikeLikeEvent |= (lV > (-4.00000000000000000e+01f));
            }
            // register a spike-like event
            if (spikeLikeEvent) {
                spkEvntIdx = atomicAdd((unsigned int *) &spkEvntCount, 1);
                shSpkEvnt[spkEvntIdx] = lid;
            }
            // test for and register a true spike
            if ((lV >= 0.0f) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
            }
            dd_VLHI[lid] = lV;
            dd_mLHI[lid] = lm;
            dd_hLHI[lid] = lh;
            dd_nLHI[lid] = ln;
            // the post-synaptic dynamics
            linSynPNLHI*=(9.04837418035959518e-01f);
            dd_inSynPNLHI[lid] = linSynPNLHI;
        }
        __syncthreads();
        if (threadIdx.x == 1) {
            if (spkEvntCount > 0) posSpkEvnt = atomicAdd((unsigned int *) &dd_glbSpkCntEvntLHI[0], spkEvntCount);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntLHI[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkEvntCount) {
            dd_glbSpkEvntLHI[posSpkEvnt + threadIdx.x] = shSpkEvnt[threadIdx.x];
        }
        if (threadIdx.x < spkCount) {
            dd_glbSpkLHI[posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
    // neuron group PN
    if ((id >= 1216) && (id < 1344)) {
        unsigned int lid = id - 1216;
        
        // only do this for existing neurons
        if (lid < 100) {
            // pull neuron variables in a coalesced access
            scalar lV = dd_VPN[lid];
            uint64_t lseed = dd_seedPN[lid];
            scalar lspikeTime = dd_spikeTimePN[lid];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (2.00000000000000000e+01f));
            // calculate membrane potential
            uint64_t theRnd;
            if (lV > (-6.00000000000000000e+01f)) {
               lV= (-6.00000000000000000e+01f);
            }else if (t - lspikeTime > ((2.50000000000000000e+00f))) {
               MYRAND(lseed,theRnd);
               if (theRnd < *(ratesPN+offsetPN+lid)) {
                   lV= (2.00000000000000000e+01f);
                   lspikeTime= t;
               }
            }
            
            // test for and register a true spike
            if ((lV >= (2.00000000000000000e+01f)) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
            }
            dd_VPN[lid] = lV;
            dd_seedPN[lid] = lseed;
            dd_spikeTimePN[lid] = lspikeTime;
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
