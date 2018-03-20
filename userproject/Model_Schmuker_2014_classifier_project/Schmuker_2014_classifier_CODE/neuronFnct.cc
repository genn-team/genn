

#ifndef _Schmuker_2014_classifier_neuronFnct_cc
#define _Schmuker_2014_classifier_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model Schmuker_2014_classifier containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group AN
     {
        glbSpkCntEvntAN[0] = 0;
        glbSpkCntAN[0] = 0;
        
        for (int n = 0; n < 180; n++) {
            scalar lV = VAN[n];
            scalar lpreV = preVAN[n];
            
            float Isyn = 0;
            Isyn += inSynPNAN[n] * ((0.00000000000000000e+00f) - lV);
            Isyn += inSynANAN[n] * ((-9.20000000000000000e+01f) - lV);
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
                glbSpkEvntAN[glbSpkCntEvntAN[0]++] = n;
                }
            // test for and register a true spike
            if ((lV >= (3.19200000000000159e+01f)) && !(oldSpike)) {
                glbSpkAN[glbSpkCntAN[0]++] = n;
                }
            VAN[n] = lV;
            preVAN[n] = lpreV;
            // the post-synaptic dynamics
            inSynPNAN[n]*=(6.06530659712633424e-01f);
            // the post-synaptic dynamics
            inSynANAN[n]*=(9.39413062813475808e-01f);
            }
        }
    
    // neuron group PN
     {
        glbSpkCntEvntPN[0] = 0;
        glbSpkCntPN[0] = 0;
        
        for (int n = 0; n < 600; n++) {
            scalar lV = VPN[n];
            scalar lpreV = preVPN[n];
            
            float Isyn = 0;
            Isyn += inSynRNPN[n] * ((0.00000000000000000e+00f) - lV);
            Isyn += inSynPNPN[n] * ((-9.20000000000000000e+01f) - lV);
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
                glbSpkEvntPN[glbSpkCntEvntPN[0]++] = n;
                }
            // test for and register a true spike
            if ((lV >= (3.19200000000000159e+01f)) && !(oldSpike)) {
                glbSpkPN[glbSpkCntPN[0]++] = n;
                }
            VPN[n] = lV;
            preVPN[n] = lpreV;
            // the post-synaptic dynamics
            inSynRNPN[n]*=(6.06530659712633424e-01f);
            // the post-synaptic dynamics
            inSynPNPN[n]*=(9.13100716282262304e-01f);
            }
        }
    
    // neuron group RN
     {
        glbSpkCntEvntRN[0] = 0;
        glbSpkCntRN[0] = 0;
        
        for (int n = 0; n < 600; n++) {
            scalar lV = VRN[n];
            uint64_t lseed = seedRN[n];
            scalar lspikeTime = spikeTimeRN[n];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= (2.00000000000000000e+01f));
            // calculate membrane potential
            uint64_t theRnd;
if (lV > (-6.00000000000000000e+01f)) {
   lV= (-6.00000000000000000e+01f);
}else if (t - lspikeTime > ((2.50000000000000000e+00f))) {
   MYRAND(lseed,theRnd);
   if (theRnd < *(ratesRN+offsetRN+n)) {
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
                glbSpkEvntRN[glbSpkCntEvntRN[0]++] = n;
                }
            // test for and register a true spike
            if ((lV >= (2.00000000000000000e+01f)) && !(oldSpike)) {
                glbSpkRN[glbSpkCntRN[0]++] = n;
                }
            VRN[n] = lV;
            seedRN[n] = lseed;
            spikeTimeRN[n] = lspikeTime;
            }
        }
    
    }

    #endif
