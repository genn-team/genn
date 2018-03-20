

#ifndef _Schmuker_2014_classifier_synapseFnct_cc
#define _Schmuker_2014_classifier_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model Schmuker_2014_classifier containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapseDynamicsCPU(float t)
 {
    // execute internal synapse dynamics if any
    }
void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    float addtoinSyn;
    
    // synapse group ANAN
     {
        // process presynaptic events: Spike type events
        for (int i = 0; i < glbSpkCntEvntAN[0]; i++) {
            ipre = glbSpkEvntAN[i];
            for (ipost = 0; ipost < 180; ipost++) {
                if (VAN[ipre] > (-3.50000000000000000e+01f)) {
                    addtoinSyn = gANAN[ipre * 180 + ipost];
inSynANAN[ipost] += addtoinSyn;

                    }
                }
            }
        }
    
    // synapse group PNAN
     {
        // process presynaptic events: Spike type events
        for (int i = 0; i < glbSpkCntEvntPN[0]; i++) {
            ipre = glbSpkEvntPN[i];
            for (ipost = 0; ipost < 180; ipost++) {
                if (VPN[ipre] > (-2.00000000000000000e+01f)) {
                    addtoinSyn = gPNAN[ipre * 180 + ipost];
inSynPNAN[ipost] += addtoinSyn;

                    }
                }
            }
        }
    
    // synapse group PNPN
     {
        // process presynaptic events: Spike type events
        for (int i = 0; i < glbSpkCntEvntPN[0]; i++) {
            ipre = glbSpkEvntPN[i];
            for (ipost = 0; ipost < 600; ipost++) {
                if (VPN[ipre] > (-3.50000000000000000e+01f)) {
                    addtoinSyn = gPNPN[ipre * 600 + ipost];
inSynPNPN[ipost] += addtoinSyn;

                    }
                }
            }
        }
    
    // synapse group RNPN
     {
        // process presynaptic events: Spike type events
        for (int i = 0; i < glbSpkCntEvntRN[0]; i++) {
            ipre = glbSpkEvntRN[i];
            for (ipost = 0; ipost < 600; ipost++) {
                if (20.000000f > (-2.00000000000000000e+01f)) {
                    addtoinSyn = gRNPN[ipre * 600 + ipost];
inSynRNPN[ipost] += addtoinSyn;

                    }
                }
            }
        }
    
    }


#endif
