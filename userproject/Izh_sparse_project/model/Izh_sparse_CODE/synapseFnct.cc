

#ifndef _Izh_sparse_synapseFnct_cc
#define _Izh_sparse_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model Izh_sparse containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    unsigned int npost;
    float addtoinSyn;
    
    // synapse group Exc_Exc
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPExc[0]; i++) {
            ipre = glbSpkPExc[i];
            npost = CExc_Exc.indInG[ipre + 1] - CExc_Exc.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CExc_Exc.ind[CExc_Exc.indInG[ipre] + j];
                addtoinSyn = gExc_Exc[CExc_Exc.indInG[ipre] + j];
                inSynExc_Exc[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group Exc_Inh
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPExc[0]; i++) {
            ipre = glbSpkPExc[i];
            npost = CExc_Inh.indInG[ipre + 1] - CExc_Inh.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CExc_Inh.ind[CExc_Inh.indInG[ipre] + j];
                addtoinSyn = gExc_Inh[CExc_Inh.indInG[ipre] + j];
                inSynExc_Inh[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group Inh_Exc
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPInh[0]; i++) {
            ipre = glbSpkPInh[i];
            npost = CInh_Exc.indInG[ipre + 1] - CInh_Exc.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CInh_Exc.ind[CInh_Exc.indInG[ipre] + j];
                addtoinSyn = gInh_Exc[CInh_Exc.indInG[ipre] + j];
                inSynInh_Exc[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group Inh_Inh
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPInh[0]; i++) {
            ipre = glbSpkPInh[i];
            npost = CInh_Inh.indInG[ipre + 1] - CInh_Inh.indInG[ipre];
            for (int j = 0; j < npost; j++) {
                ipost = CInh_Inh.ind[CInh_Inh.indInG[ipre] + j];
                addtoinSyn = gInh_Inh[CInh_Inh.indInG[ipre] + j];
                inSynInh_Inh[ipost] += addtoinSyn;
                
            }
        }
    }
    
}


#endif
