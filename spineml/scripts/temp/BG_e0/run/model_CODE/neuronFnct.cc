

#ifndef _model_neuronFnct_cc
#define _model_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model model containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group Cortex
     {
        glbSpkCntCortex[0] = 0;
        
        for (int n = 0; n < 6; n++) {
            scalar la = aCortex[n];
            scalar lin = inCortex[n];
            scalar lout = outCortex[n];
            
            // calculate membrane potential
            if(la < -(0.00000000000000000e+00f)) {
                la = la=-(0.00000000000000000e+00f);
            }
            la += DT * ((-la+lin)/(1.00000000000000006e-01f));
            lout = (1.00000000000000000e+00f)*la+(0.00000000000000000e+00f);
            
            aCortex[n] = la;
            inCortex[n] = lin;
            outCortex[n] = lout;
        }
    }
    
    // neuron group D1
     {
        glbSpkCntD1[0] = 0;
        
        for (int n = 0; n < 6; n++) {
            scalar la = aD1[n];
            scalar lout = outD1[n];
            
            float Isyn = 0;
            scalar in = 0;
            in += inSynCortex_to_D1_Synapse_0_weight_update[n];inSynCortex_to_D1_Synapse_0_weight_update[n] = 0;
            
            // calculate membrane potential
            if(la<-(-2.00000000000000011e-01f)) {
                la = -(-2.00000000000000011e-01f);
            }
            la += DT * ((-la+in*(1+(2.00000000000000011e-01f)))/(2.50000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(-2.00000000000000011e-01f);
            
            aD1[n] = la;
            outD1[n] = lout;
            // the post-synaptic dynamics
            
        }
    }
    
    // neuron group D2
     {
        glbSpkCntD2[0] = 0;
        
        for (int n = 0; n < 6; n++) {
            scalar la = aD2[n];
            scalar lout = outD2[n];
            
            float Isyn = 0;
            scalar in = 0;
            in += inSynCortex_to_D2_Synapse_0_weight_update[n];inSynCortex_to_D2_Synapse_0_weight_update[n] = 0;
            
            // calculate membrane potential
            if(la<-(-2.00000000000000011e-01f)) {
                la = -(-2.00000000000000011e-01f);
            }
            la += DT * ((-la+in*(1+(-2.00000000000000011e-01f)))/(2.50000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(-2.00000000000000011e-01f);
            
            aD2[n] = la;
            outD2[n] = lout;
            // the post-synaptic dynamics
            
        }
    }
    
    // neuron group GPe
     {
        spkQuePtrGPe = (spkQuePtrGPe + 1) % 2;
        glbSpkCntGPe[0] = 0;
        unsigned int delaySlot = (spkQuePtrGPe + 1) % 2;
        
        for (int n = 0; n < 6; n++) {
            scalar la = aGPe[n];
            scalar lout = outGPe[(delaySlot * 6) + n];
            
            float Isyn = 0;
            scalar in = 0;
            in += inSynD2_to_GPe_Synapse_0_weight_update[n];inSynD2_to_GPe_Synapse_0_weight_update[n] = 0;
            
            in += inSynSTN_to_GPe_Synapse_0_weight_update[n];inSynSTN_to_GPe_Synapse_0_weight_update[n] = 0;
            
            // calculate membrane potential
            if(la < -(2.00000000000000011e-01f)) {
                la = la=-(2.00000000000000011e-01f);
            }
            la += DT * ((-la+in)/(2.50000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(2.00000000000000011e-01f);
            
            aGPe[n] = la;
            outGPe[(spkQuePtrGPe * 6) + n] = lout;
            // the post-synaptic dynamics
            
            // the post-synaptic dynamics
            
        }
    }
    
    // neuron group GPi
     {
        glbSpkCntGPi[0] = 0;
        
        for (int n = 0; n < 6; n++) {
            scalar la = aGPi[n];
            scalar lout = outGPi[n];
            
            float Isyn = 0;
            scalar in = 0;
            in += inSynD1_to_GPi_Synapse_0_weight_update[n];inSynD1_to_GPi_Synapse_0_weight_update[n] = 0;
            
            in += inSynSTN_to_GPi_Synapse_0_weight_update[n];inSynSTN_to_GPi_Synapse_0_weight_update[n] = 0;
            
            in += inSynGPe_to_GPi_Synapse_0_weight_update[n];inSynGPe_to_GPi_Synapse_0_weight_update[n] = 0;
            
            // calculate membrane potential
            if(la < -(2.00000000000000011e-01f)) {
                la = la=-(2.00000000000000011e-01f);
            }
            la += DT * ((-la+in)/(2.00000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(2.00000000000000011e-01f);
            
            aGPi[n] = la;
            outGPi[n] = lout;
            // the post-synaptic dynamics
            
            // the post-synaptic dynamics
            
            // the post-synaptic dynamics
            
        }
    }
    
    // neuron group STN
     {
        glbSpkCntSTN[0] = 0;
        
        for (int n = 0; n < 6; n++) {
            scalar la = aSTN[n];
            scalar lout = outSTN[n];
            
            float Isyn = 0;
            scalar in = 0;
            in += inSynGPe_to_STN_Synapse_0_weight_update[n];inSynGPe_to_STN_Synapse_0_weight_update[n] = 0;
            
            in += inSynCortex_to_STN_Synapse_0_weight_update[n];inSynCortex_to_STN_Synapse_0_weight_update[n] = 0;
            
            // calculate membrane potential
            if(la < -(2.50000000000000000e-01f)) {
                la = la=-(2.50000000000000000e-01f);
            }
            la += DT * ((-la+in)/(2.50000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(2.50000000000000000e-01f);
            
            aSTN[n] = la;
            outSTN[n] = lout;
            // the post-synaptic dynamics
            
            // the post-synaptic dynamics
            
        }
    }
    
}

#endif
