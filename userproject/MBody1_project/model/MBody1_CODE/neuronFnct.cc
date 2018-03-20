

#ifndef _MBody1_neuronFnct_cc
#define _MBody1_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model MBody1 containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group DN
     {
        glbSpkCntEvntDN[0] = 0;
        glbSpkCntDN[0] = 0;
        
        for (int n = 0; n < 100; n++) {
            scalar lV = VDN[n];
            scalar lm = mDN[n];
            scalar lh = hDN[n];
            scalar ln = nDN[n];
            
            float Isyn = 0;
            Isyn += inSynKCDN[n] * ((0.00000000000000000e+00f) - lV);
            Isyn += inSynDNDN[n] * ((-9.20000000000000000e+01f) - lV);
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
                glbSpkEvntDN[glbSpkCntEvntDN[0]++] = n;
            }
            // test for and register a true spike
            if ((lV >= 0.0f) && !(oldSpike)) {
                glbSpkDN[glbSpkCntDN[0]++] = n;
                sTDN[n] = t;
            }
            VDN[n] = lV;
            mDN[n] = lm;
            hDN[n] = lh;
            nDN[n] = ln;
            // the post-synaptic dynamics
            inSynKCDN[n]*=(9.80198673306755253e-01f);
            // the post-synaptic dynamics
            inSynDNDN[n]*=(9.60789439152323177e-01f);
        }
    }
    
    // neuron group KC
     {
        glbSpkCntKC[0] = 0;
        
        for (int n = 0; n < 1000; n++) {
            scalar lV = VKC[n];
            scalar lm = mKC[n];
            scalar lh = hKC[n];
            scalar ln = nKC[n];
            
            float Isyn = 0;
            Isyn += inSynPNKC[n] * ((0.00000000000000000e+00f) - lV);
            Isyn += inSynLHIKC[n] * ((-9.20000000000000000e+01f) - lV);
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
            if ((lV >= 0.0f) && !(oldSpike)) {
                glbSpkKC[glbSpkCntKC[0]++] = n;
                sTKC[n] = t;
            }
            VKC[n] = lV;
            mKC[n] = lm;
            hKC[n] = lh;
            nKC[n] = ln;
            // the post-synaptic dynamics
            inSynPNKC[n]*=(9.04837418035959518e-01f);
            // the post-synaptic dynamics
            inSynLHIKC[n]*=(9.35506985031617777e-01f);
        }
    }
    
    // neuron group LHI
     {
        glbSpkCntEvntLHI[0] = 0;
        glbSpkCntLHI[0] = 0;
        
        for (int n = 0; n < 20; n++) {
            scalar lV = VLHI[n];
            scalar lm = mLHI[n];
            scalar lh = hLHI[n];
            scalar ln = nLHI[n];
            
            float Isyn = 0;
            Isyn += inSynPNLHI[n] * ((0.00000000000000000e+00f) - lV);
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
                glbSpkEvntLHI[glbSpkCntEvntLHI[0]++] = n;
            }
            // test for and register a true spike
            if ((lV >= 0.0f) && !(oldSpike)) {
                glbSpkLHI[glbSpkCntLHI[0]++] = n;
            }
            VLHI[n] = lV;
            mLHI[n] = lm;
            hLHI[n] = lh;
            nLHI[n] = ln;
            // the post-synaptic dynamics
            inSynPNLHI[n]*=(9.04837418035959518e-01f);
        }
    }
    
    // neuron group PN
     {
        glbSpkCntPN[0] = 0;
        
        for (int n = 0; n < 100; n++) {
            scalar lV = VPN[n];
            uint64_t lseed = seedPN[n];
            scalar lspikeTime = spikeTimePN[n];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (2.00000000000000000e+01f));
            // calculate membrane potential
            uint64_t theRnd;
            if (lV > (-6.00000000000000000e+01f)) {
               lV= (-6.00000000000000000e+01f);
            }else if (t - lspikeTime > ((2.50000000000000000e+00f))) {
               MYRAND(lseed,theRnd);
               if (theRnd < *(ratesPN+offsetPN+n)) {
                   lV= (2.00000000000000000e+01f);
                   lspikeTime= t;
               }
            }
            
            // test for and register a true spike
            if ((lV >= (2.00000000000000000e+01f)) && !(oldSpike)) {
                glbSpkPN[glbSpkCntPN[0]++] = n;
            }
            VPN[n] = lV;
            seedPN[n] = lseed;
            spikeTimePN[n] = lspikeTime;
        }
    }
    
}

#endif
