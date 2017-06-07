

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
        
        for (int n = 0; n < 10; n++) {
            scalar lV = VDN[n];
            scalar lpreV = preVDN[n];
            
            float Isyn = 0;
            Isyn += inSynKCDN[n] * ((0.00000000000000000e+00f) - lV);
            Isyn += inSynDNDN[n] * ((-9.20000000000000000e+01f) - lV);
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (3.19200000000000159e+01f));
            // calculate membrane potential
            if (lV <= 0) {
   lpreV= lV;
   lV= (1.08000000000000000e+04f)/(((6.00000000000000000e+01f)) - lV - ((2.64000000000000012e+00f))*Isyn) +((-1.48079999999999984e+02f));
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
                spikeLikeEvent |= (lV > (-3.00000000000000000e+01f));
                }
            // register a spike-like event
            if (spikeLikeEvent) {
                glbSpkEvntDN[glbSpkCntEvntDN[0]++] = n;
                }
            // test for and register a true spike
            if ((lV >= (3.19200000000000159e+01f)) && !(oldSpike)) {
                glbSpkDN[glbSpkCntDN[0]++] = n;
                sTDN[n] = t;
                }
            VDN[n] = lV;
            preVDN[n] = lpreV;
            // the post-synaptic dynamics
            inSynKCDN[n]*=(9.04837418035959518e-01f);
            // the post-synaptic dynamics
            inSynDNDN[n]*=(8.18730753077981821e-01f);
            }
        }
    
    // neuron group KC
     {
        glbSpkCntKC[0] = 0;
        
        for (int n = 0; n < 5000; n++) {
            scalar lV = VKC[n];
            scalar lpreV = preVKC[n];
            
            float Isyn = 0;
            Isyn += inSynPNKC[n] * ((0.00000000000000000e+00f) - lV);
            Isyn += inSynLHIKC[n] * ((-9.20000000000000000e+01f) - lV);
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (3.19200000000000159e+01f));
            // calculate membrane potential
            if (lV <= 0) {
   lpreV= lV;
   lV= (1.08000000000000000e+04f)/(((6.00000000000000000e+01f)) - lV - ((2.64000000000000012e+00f))*Isyn) +((-1.48079999999999984e+02f));
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

            // test for and register a true spike
            if ((lV >= (3.19200000000000159e+01f)) && !(oldSpike)) {
                glbSpkKC[glbSpkCntKC[0]++] = n;
                sTKC[n] = t;
                }
            VKC[n] = lV;
            preVKC[n] = lpreV;
            // the post-synaptic dynamics
            inSynPNKC[n]*=(6.06530659712633424e-01f);
            // the post-synaptic dynamics
            inSynLHIKC[n]*=(7.16531310573789271e-01f);
            }
        }
    
    // neuron group LHI
     {
        glbSpkCntEvntLHI[0] = 0;
        glbSpkCntLHI[0] = 0;
        
        for (int n = 0; n < 20; n++) {
            scalar lV = VLHI[n];
            scalar lpreV = preVLHI[n];
            
            float Isyn = 0;
            Isyn += inSynPNLHI[n] * ((0.00000000000000000e+00f) - lV);
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (3.19200000000000159e+01f));
            // calculate membrane potential
            if (lV <= 0) {
   lpreV= lV;
   lV= (1.08000000000000000e+04f)/(((6.00000000000000000e+01f)) - lV - ((2.64000000000000012e+00f))*Isyn) +((-1.48079999999999984e+02f));
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
                spikeLikeEvent |= (lV > (-4.00000000000000000e+01f));
                }
            // register a spike-like event
            if (spikeLikeEvent) {
                glbSpkEvntLHI[glbSpkCntEvntLHI[0]++] = n;
                }
            // test for and register a true spike
            if ((lV >= (3.19200000000000159e+01f)) && !(oldSpike)) {
                glbSpkLHI[glbSpkCntLHI[0]++] = n;
                }
            VLHI[n] = lV;
            preVLHI[n] = lpreV;
            // the post-synaptic dynamics
            inSynPNLHI[n]*=(6.06530659712633424e-01f);
            }
        }
    
    // neuron group PN
     {
        glbSpkCntPN[0] = 0;
        
        for (int n = 0; n < 1024; n++) {
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
