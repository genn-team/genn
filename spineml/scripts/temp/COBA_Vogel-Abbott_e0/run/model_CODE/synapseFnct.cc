

#ifndef _model_synapseFnct_cc
#define _model_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model model containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    unsigned int npost;
    float addtoinSyn;
    
    // synapse group Excitatory_to_Excitatory_Synapse_0_weight_update
     {
        unsigned int delaySlot = (spkQuePtrExcitatory + 1) % 2;
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntExcitatory[delaySlot]; i++) {
            ipre = glbSpkExcitatory[(delaySlot * 3200) + i];
            npost = CExcitatory_to_Excitatory_Synapse_0_weight_update.indInG[ipre + 1] - CExcitatory_to_Excitatory_Synapse_0_weight_update.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CExcitatory_to_Excitatory_Synapse_0_weight_update.ind[CExcitatory_to_Excitatory_Synapse_0_weight_update.indInG[ipre] + j];
                addtoinSyn = (4.00000000000000008e-03f);
                inSynExcitatory_to_Excitatory_Synapse_0_weight_update[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group Excitatory_to_Inhibitory_Synapse_0_weight_update
     {
        unsigned int delaySlot = (spkQuePtrExcitatory + 1) % 2;
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntExcitatory[delaySlot]; i++) {
            ipre = glbSpkExcitatory[(delaySlot * 3200) + i];
            npost = CExcitatory_to_Inhibitory_Synapse_0_weight_update.indInG[ipre + 1] - CExcitatory_to_Inhibitory_Synapse_0_weight_update.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CExcitatory_to_Inhibitory_Synapse_0_weight_update.ind[CExcitatory_to_Inhibitory_Synapse_0_weight_update.indInG[ipre] + j];
                addtoinSyn = (4.00000000000000008e-03f);
                inSynExcitatory_to_Inhibitory_Synapse_0_weight_update[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group Inhibitory_to_Excitatory_Synapse_0_weight_update
     {
        unsigned int delaySlot = (spkQuePtrInhibitory + 1) % 2;
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntInhibitory[delaySlot]; i++) {
            ipre = glbSpkInhibitory[(delaySlot * 800) + i];
            npost = CInhibitory_to_Excitatory_Synapse_0_weight_update.indInG[ipre + 1] - CInhibitory_to_Excitatory_Synapse_0_weight_update.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CInhibitory_to_Excitatory_Synapse_0_weight_update.ind[CInhibitory_to_Excitatory_Synapse_0_weight_update.indInG[ipre] + j];
                addtoinSyn = (5.09999999999999967e-02f);
                inSynInhibitory_to_Excitatory_Synapse_0_weight_update[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group Inhibitory_to_Inhibitory_Synapse_0_weight_update
     {
        unsigned int delaySlot = (spkQuePtrInhibitory + 1) % 2;
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntInhibitory[delaySlot]; i++) {
            ipre = glbSpkInhibitory[(delaySlot * 800) + i];
            npost = CInhibitory_to_Inhibitory_Synapse_0_weight_update.indInG[ipre + 1] - CInhibitory_to_Inhibitory_Synapse_0_weight_update.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CInhibitory_to_Inhibitory_Synapse_0_weight_update.ind[CInhibitory_to_Inhibitory_Synapse_0_weight_update.indInG[ipre] + j];
                addtoinSyn = (5.09999999999999967e-02f);
                inSynInhibitory_to_Inhibitory_Synapse_0_weight_update[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group Spike_Source_to_Excitatory_Synapse_0_weight_update
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntSpike_Source[0]; i++) {
            ipre = glbSpkSpike_Source[i];
            npost = CSpike_Source_to_Excitatory_Synapse_0_weight_update.indInG[ipre + 1] - CSpike_Source_to_Excitatory_Synapse_0_weight_update.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CSpike_Source_to_Excitatory_Synapse_0_weight_update.ind[CSpike_Source_to_Excitatory_Synapse_0_weight_update.indInG[ipre] + j];
                addtoinSyn = (1.00000000000000006e-01f);
                inSynSpike_Source_to_Excitatory_Synapse_0_weight_update[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group Spike_Source_to_Inhibitory_Synapse_0_weight_update
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntSpike_Source[0]; i++) {
            ipre = glbSpkSpike_Source[i];
            npost = CSpike_Source_to_Inhibitory_Synapse_0_weight_update.indInG[ipre + 1] - CSpike_Source_to_Inhibitory_Synapse_0_weight_update.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CSpike_Source_to_Inhibitory_Synapse_0_weight_update.ind[CSpike_Source_to_Inhibitory_Synapse_0_weight_update.indInG[ipre] + j];
                addtoinSyn = (1.00000000000000006e-01f);
                inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update[ipost] += addtoinSyn;
                
            }
        }
    }
    
}


#endif
