

#ifndef _PoissonIzh_neuronFnct_cc
#define _PoissonIzh_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model PoissonIzh containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group Izh1
     {
        glbSpkCntIzh1[0] = 0;
        
        for (int n = 0; n < 10; n++) {
            scalar lV = VIzh1[n];
            scalar lU = UIzh1[n];
            
            float Isyn = 0;
            Isyn += inSynPNIzh1[n]; inSynPNIzh1[n] = 0;
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
                glbSpkIzh1[glbSpkCntIzh1[0]++] = n;
            }
            VIzh1[n] = lV;
            UIzh1[n] = lU;
            // the post-synaptic dynamics
            
        }
    }
    
    // neuron group PN
     {
        glbSpkCntPN[0] = 0;
        
        for (int n = 0; n < 100; n++) {
            scalar ltimeStepToSpike = timeStepToSpikePN[n];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= (ltimeStepToSpike <= 0.0f);
            // calculate membrane potential
            if(ltimeStepToSpike <= 0.0f) {
                ltimeStepToSpike += (5.00000000000000000e+01f) * standardExponentialDistribution(rng);
            }
            ltimeStepToSpike -= 1.0f;
            
            // test for and register a true spike
            if ((ltimeStepToSpike <= 0.0f) && !(oldSpike)) {
                glbSpkPN[glbSpkCntPN[0]++] = n;
            }
            timeStepToSpikePN[n] = ltimeStepToSpike;
        }
    }
    
}

#endif
