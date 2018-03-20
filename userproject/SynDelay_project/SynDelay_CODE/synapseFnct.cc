

#ifndef _SynDelay_synapseFnct_cc
#define _SynDelay_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model SynDelay containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    float addtoinSyn;
    
    // synapse group InputInter
     {
        unsigned int delaySlot = (spkQuePtrInput + 4) % 7;
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntInput[delaySlot]; i++) {
            ipre = glbSpkInput[(delaySlot * 500) + i];
            for (ipost = 0; ipost < 500; ipost++) {
                addtoinSyn = (5.99999999999999978e-02f);
                inSynInputInter[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group InputOutput
     {
        unsigned int delaySlot = (spkQuePtrInput + 1) % 7;
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntInput[delaySlot]; i++) {
            ipre = glbSpkInput[(delaySlot * 500) + i];
            for (ipost = 0; ipost < 500; ipost++) {
                addtoinSyn = (2.99999999999999989e-02f);
                inSynInputOutput[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group InterOutput
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntInter[0]; i++) {
            ipre = glbSpkInter[i];
            for (ipost = 0; ipost < 500; ipost++) {
                addtoinSyn = (2.99999999999999989e-02f);
                inSynInterOutput[ipost] += addtoinSyn;
                
            }
        }
    }
    
}


#endif
