

#ifndef _SynDelay_neuronFnct_cc
#define _SynDelay_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model SynDelay containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group Input
     {
        spkQuePtrInput = (spkQuePtrInput + 1) % 7;
        glbSpkCntInput[spkQuePtrInput] = 0;
        
        for (int n = 0; n < 500; n++) {
            scalar lV = VInput[n];
            scalar lU = UInput[n];
            
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
            if ((lV >= 29.99f) && !(oldSpike)) {
                glbSpkInput[(spkQuePtrInput * 500) + glbSpkCntInput[spkQuePtrInput]++] = n;
            }
            VInput[n] = lV;
            UInput[n] = lU;
        }
    }
    
    // neuron group Inter
     {
        glbSpkCntInter[0] = 0;
        
        for (int n = 0; n < 500; n++) {
            scalar lV = VInter[n];
            scalar lU = UInter[n];
            
            float Isyn = 0;
            Isyn += inSynInputInter[n] * ((0.00000000000000000e+00f) - lV);
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
            if ((lV >= 29.99f) && !(oldSpike)) {
                glbSpkInter[glbSpkCntInter[0]++] = n;
            }
            VInter[n] = lV;
            UInter[n] = lU;
            // the post-synaptic dynamics
            inSynInputInter[n]*=(3.67879441171442334e-01f);
        }
    }
    
    // neuron group Output
     {
        glbSpkCntOutput[0] = 0;
        
        for (int n = 0; n < 500; n++) {
            scalar lV = VOutput[n];
            scalar lU = UOutput[n];
            
            float Isyn = 0;
            Isyn += inSynInputOutput[n] * ((0.00000000000000000e+00f) - lV);
            Isyn += inSynInterOutput[n] * ((0.00000000000000000e+00f) - lV);
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
            if ((lV >= 29.99f) && !(oldSpike)) {
                glbSpkOutput[glbSpkCntOutput[0]++] = n;
            }
            VOutput[n] = lV;
            UOutput[n] = lU;
            // the post-synaptic dynamics
            inSynInputOutput[n]*=(3.67879441171442334e-01f);
            // the post-synaptic dynamics
            inSynInterOutput[n]*=(3.67879441171442334e-01f);
        }
    }
    
}

#endif
