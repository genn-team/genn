

#ifndef _PoissonIzh_synapseFnct_cc
#define _PoissonIzh_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model PoissonIzh containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    float addtoinSyn;
    
    // synapse group PNIzh1
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPN[0]; i++) {
            ipre = glbSpkPN[i];
            for (ipost = 0; ipost < 10; ipost++) {
                addtoinSyn = gPNIzh1[ipre * 10 + ipost];
                inSynPNIzh1[ipost] += addtoinSyn;
                
            }
        }
    }
    
}


#endif
