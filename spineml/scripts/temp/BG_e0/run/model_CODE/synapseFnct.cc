

#ifndef _model_synapseFnct_cc
#define _model_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model model containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapseDynamicsCPU(float t)
 {
    float addtoinSyn;
    
    // execute internal synapse dynamics if any
    // synapse group Cortex_to_D1_Synapse_0_weight_update
     {
        for (int n= 0; n < CCortex_to_D1_Synapse_0_weight_update.connN; n++) {
            
            addtoinSyn = (1.00000000000000000e+00f)*outCortex[CCortex_to_D1_Synapse_0_weight_update.preInd[n]];
            inSynCortex_to_D1_Synapse_0_weight_update[CCortex_to_D1_Synapse_0_weight_update.ind[n]] += addtoinSyn;
            
        }
    }
    // synapse group Cortex_to_D2_Synapse_0_weight_update
     {
        for (int n= 0; n < CCortex_to_D2_Synapse_0_weight_update.connN; n++) {
            
            addtoinSyn = (1.00000000000000000e+00f)*outCortex[CCortex_to_D2_Synapse_0_weight_update.preInd[n]];
            inSynCortex_to_D2_Synapse_0_weight_update[CCortex_to_D2_Synapse_0_weight_update.ind[n]] += addtoinSyn;
            
        }
    }
    // synapse group Cortex_to_STN_Synapse_0_weight_update
     {
        for (int n= 0; n < CCortex_to_STN_Synapse_0_weight_update.connN; n++) {
            
            addtoinSyn = (1.00000000000000000e+00f)*outCortex[CCortex_to_STN_Synapse_0_weight_update.preInd[n]];
            inSynCortex_to_STN_Synapse_0_weight_update[CCortex_to_STN_Synapse_0_weight_update.ind[n]] += addtoinSyn;
            
        }
    }
    // synapse group D1_to_GPi_Synapse_0_weight_update
     {
        for (int n= 0; n < CD1_to_GPi_Synapse_0_weight_update.connN; n++) {
            
            addtoinSyn = (-1.00000000000000000e+00f)*outD1[CD1_to_GPi_Synapse_0_weight_update.preInd[n]];
            inSynD1_to_GPi_Synapse_0_weight_update[CD1_to_GPi_Synapse_0_weight_update.ind[n]] += addtoinSyn;
            
        }
    }
    // synapse group D2_to_GPe_Synapse_0_weight_update
     {
        for (int n= 0; n < CD2_to_GPe_Synapse_0_weight_update.connN; n++) {
            
            addtoinSyn = (-1.00000000000000000e+00f)*outD2[CD2_to_GPe_Synapse_0_weight_update.preInd[n]];
            inSynD2_to_GPe_Synapse_0_weight_update[CD2_to_GPe_Synapse_0_weight_update.ind[n]] += addtoinSyn;
            
        }
    }
    // synapse group GPe_to_GPi_Synapse_0_weight_update
     {
        unsigned int delaySlot = (spkQuePtrGPe + 2) % 2;
        for (int n= 0; n < CGPe_to_GPi_Synapse_0_weight_update.connN; n++) {
            
            addtoinSyn = (-4.00000000000000022e-01f)*outGPe[(delaySlot * 6) + CGPe_to_GPi_Synapse_0_weight_update.preInd[n]];
            inSynGPe_to_GPi_Synapse_0_weight_update[CGPe_to_GPi_Synapse_0_weight_update.ind[n]] += addtoinSyn;
            
        }
    }
    // synapse group GPe_to_STN_Synapse_0_weight_update
     {
        unsigned int delaySlot = (spkQuePtrGPe + 1) % 2;
        for (int n= 0; n < CGPe_to_STN_Synapse_0_weight_update.connN; n++) {
            
            addtoinSyn = (-1.00000000000000000e+00f)*outGPe[(delaySlot * 6) + CGPe_to_STN_Synapse_0_weight_update.preInd[n]];
            inSynGPe_to_STN_Synapse_0_weight_update[CGPe_to_STN_Synapse_0_weight_update.ind[n]] += addtoinSyn;
            
        }
    }
    // synapse group STN_to_GPe_Synapse_0_weight_update
     {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                // loop through all synapses
                addtoinSyn = (9.00000000000000022e-01f)*outSTN[i];
                inSynSTN_to_GPe_Synapse_0_weight_update[j] += addtoinSyn;
                
            }
        }
    }
    // synapse group STN_to_GPi_Synapse_0_weight_update
     {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                // loop through all synapses
                addtoinSyn = (9.00000000000000022e-01f)*outSTN[i];
                inSynSTN_to_GPi_Synapse_0_weight_update[j] += addtoinSyn;
                
            }
        }
    }
}
void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    unsigned int npost;
    float addtoinSyn;
    
    // synapse group Cortex_to_D1_Synapse_0_weight_update
     {
    }
    
    // synapse group Cortex_to_D2_Synapse_0_weight_update
     {
    }
    
    // synapse group Cortex_to_STN_Synapse_0_weight_update
     {
    }
    
    // synapse group D1_to_GPi_Synapse_0_weight_update
     {
    }
    
    // synapse group D2_to_GPe_Synapse_0_weight_update
     {
    }
    
    // synapse group GPe_to_GPi_Synapse_0_weight_update
     {
        unsigned int delaySlot = (spkQuePtrGPe + 2) % 2;
    }
    
    // synapse group GPe_to_STN_Synapse_0_weight_update
     {
        unsigned int delaySlot = (spkQuePtrGPe + 1) % 2;
    }
    
    // synapse group STN_to_GPe_Synapse_0_weight_update
     {
    }
    
    // synapse group STN_to_GPi_Synapse_0_weight_update
     {
    }
    
}


#endif
