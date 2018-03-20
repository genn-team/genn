

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
    // neuron group Excitatory
     {
        spkQuePtrExcitatory = (spkQuePtrExcitatory + 1) % 2;
        glbSpkCntExcitatory[spkQuePtrExcitatory] = 0;
        
        for (int n = 0; n < 3200; n++) {
            scalar lt_spike = t_spikeExcitatory[n];
            scalar lv = vExcitatory[n];
            unsigned int l_regimeID = _regimeIDExcitatory[n];
            
            float Isyn = 0;
            scalar I_syn = 0;
            I_syn += inSynExcitatory_to_Excitatory_Synapse_0_weight_update[n]*((0.00000000000000000e+00f)-lv);
            
            I_syn += inSynInhibitory_to_Excitatory_Synapse_0_weight_update[n]*((-8.00000000000000000e+01f)-lv);
            
            I_syn += inSynSpike_Source_to_Excitatory_Synapse_0_weight_update[n]*((0.00000000000000000e+00f)-lv);
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if(l_regimeID == 0) {
                if(lv > (-5.00000000000000000e+01f)) {
                    lv = (-6.00000000000000000e+01f);
                    lt_spike = t;
                    l_regimeID = 1;
                }
                lv += DT * ((((0.00000000000000000e+00f) + I_syn) / (2.00000000000000011e-01f)) + ((-6.00000000000000000e+01f) - lv) / (2.00000000000000000e+01f));
            }
            else if(l_regimeID == 1) {
                if(t > (lt_spike + (5.00000000000000000e+00f))) {
                    l_regimeID = 0;
                }
            }
            
            // test for and register a true spike
            if ((l_regimeID == 0 && (lv > (-5.00000000000000000e+01f))))  {
                glbSpkExcitatory[(spkQuePtrExcitatory * 3200) + glbSpkCntExcitatory[spkQuePtrExcitatory]++] = n;
            }
            t_spikeExcitatory[n] = lt_spike;
            vExcitatory[n] = lv;
            _regimeIDExcitatory[n] = l_regimeID;
            // the post-synaptic dynamics
            inSynExcitatory_to_Excitatory_Synapse_0_weight_update[n] *=  (9.80198673306755253e-01f);
            
            // the post-synaptic dynamics
            inSynInhibitory_to_Excitatory_Synapse_0_weight_update[n] *=  (9.80198673306755253e-01f);
            
            // the post-synaptic dynamics
            inSynSpike_Source_to_Excitatory_Synapse_0_weight_update[n] *=  (9.80198673306755253e-01f);
            
        }
    }
    
    // neuron group Inhibitory
     {
        spkQuePtrInhibitory = (spkQuePtrInhibitory + 1) % 2;
        glbSpkCntInhibitory[spkQuePtrInhibitory] = 0;
        
        for (int n = 0; n < 800; n++) {
            scalar lt_spike = t_spikeInhibitory[n];
            scalar lv = vInhibitory[n];
            unsigned int l_regimeID = _regimeIDInhibitory[n];
            
            float Isyn = 0;
            scalar I_syn = 0;
            I_syn += inSynExcitatory_to_Inhibitory_Synapse_0_weight_update[n]*((0.00000000000000000e+00f)-lv);
            
            I_syn += inSynInhibitory_to_Inhibitory_Synapse_0_weight_update[n]*((-8.00000000000000000e+01f)-lv);
            
            I_syn += inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update[n]*((0.00000000000000000e+00f)-lv);
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if(l_regimeID == 0) {
                if(lv > (-5.00000000000000000e+01f)) {
                    lv = (-6.00000000000000000e+01f);
                    lt_spike = t;
                    l_regimeID = 1;
                }
                lv += DT * ((((0.00000000000000000e+00f) + I_syn) / (2.00000000000000011e-01f)) + ((-6.00000000000000000e+01f) - lv) / (2.00000000000000000e+01f));
            }
            else if(l_regimeID == 1) {
                if(t > (lt_spike + (5.00000000000000000e+00f))) {
                    l_regimeID = 0;
                }
            }
            
            // test for and register a true spike
            if ((l_regimeID == 0 && (lv > (-5.00000000000000000e+01f))))  {
                glbSpkInhibitory[(spkQuePtrInhibitory * 800) + glbSpkCntInhibitory[spkQuePtrInhibitory]++] = n;
            }
            t_spikeInhibitory[n] = lt_spike;
            vInhibitory[n] = lv;
            _regimeIDInhibitory[n] = l_regimeID;
            // the post-synaptic dynamics
            inSynExcitatory_to_Inhibitory_Synapse_0_weight_update[n] *=  (9.80198673306755253e-01f);
            
            // the post-synaptic dynamics
            inSynInhibitory_to_Inhibitory_Synapse_0_weight_update[n] *=  (9.80198673306755253e-01f);
            
            // the post-synaptic dynamics
            inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update[n] *=  (9.80198673306755253e-01f);
            
        }
    }
    
    // neuron group Spike_Source
     {
        glbSpkCntSpike_Source[0] = 0;
        
        for (int n = 0; n < 20; n++) {
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            // test for and register a true spike
            if (0)  {
                glbSpkSpike_Source[glbSpkCntSpike_Source[0]++] = n;
            }
        }
    }
    
}

#endif
