

#ifndef _MBody1_synapseFnct_cc
#define _MBody1_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model MBody1 containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    float addtoinSyn;
    
    // synapse group DNDN
     {
        // process presynaptic events: Spike type events
        for (int i = 0; i < glbSpkCntEvntDN[0]; i++) {
            ipre = glbSpkEvntDN[i];
            for (ipost = 0; ipost < 100; ipost++) {
                if (VDN[ipre] > (-3.00000000000000000e+01f)) {
                    addtoinSyn = (5.00000000000000028e-02f) * tanhf((VDN[ipre] - (-3.00000000000000000e+01f)) / (5.00000000000000000e+01f))* DT;
                    if (addtoinSyn < 0) addtoinSyn = 0.0f;
                    inSynDNDN[ipost] += addtoinSyn;
                    
                }
            }
        }
    }
    
    // synapse group KCDN
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntKC[0]; i++) {
            ipre = glbSpkKC[i];
            for (ipost = 0; ipost < 100; ipost++) {
                addtoinSyn = gKCDN[ipre * 100 + ipost];inSynKCDN[ipost] += addtoinSyn; 
                scalar dt = sTDN[ipost] - t - ((1.00000000000000000e+01f)); 
                scalar dg = 0;
                if (dt > (3.12500000000000000e+01f))  
                    dg = -((7.49999999999999934e-05f)) ; 
                else if (dt > 0)  
                    dg = (-1.20000000000000003e-05f) * dt + ((2.99999999999999974e-04f)); 
                else if (dt > (-2.50124999999999993e+01f))  
                    dg = (1.20000000000000003e-05f) * dt + ((2.99999999999999974e-04f)); 
                else dg = - ((1.49999999999999993e-07f)) ; 
                gRawKCDN[ipre * 100 + ipost] += dg; 
                gKCDN[ipre * 100 + ipost]=(1.49999999999999994e-02f)/2 *(tanhf((3.33299999999999983e+01f)*(gRawKCDN[ipre * 100 + ipost] - ((7.49999999999999972e-03f))))+1); 
                
            }
        }
    }
    
    // synapse group LHIKC
     {
        // process presynaptic events: Spike type events
        for (int i = 0; i < glbSpkCntEvntLHI[0]; i++) {
            ipre = glbSpkEvntLHI[i];
            for (ipost = 0; ipost < 1000; ipost++) {
                if (VLHI[ipre] > (-4.00000000000000000e+01f)) {
                    addtoinSyn = (5.00000000000000028e-02f) * tanhf((VLHI[ipre] - (-4.00000000000000000e+01f)) / (5.00000000000000000e+01f))* DT;
                    if (addtoinSyn < 0) addtoinSyn = 0.0f;
                    inSynLHIKC[ipost] += addtoinSyn;
                    
                }
            }
        }
    }
    
    // synapse group PNKC
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPN[0]; i++) {
            ipre = glbSpkPN[i];
            for (ipost = 0; ipost < 1000; ipost++) {
                addtoinSyn = gPNKC[ipre * 1000 + ipost];
                inSynPNKC[ipost] += addtoinSyn;
                
            }
        }
    }
    
    // synapse group PNLHI
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntPN[0]; i++) {
            ipre = glbSpkPN[i];
            for (ipost = 0; ipost < 20; ipost++) {
                addtoinSyn = gPNLHI[ipre * 20 + ipost];
                inSynPNLHI[ipost] += addtoinSyn;
                
            }
        }
    }
    
}

void learnSynapsesPostHost(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    unsigned int lSpk;
    
    // synapse group KCDN
     {
        for (ipost = 0; ipost < glbSpkCntDN[0]; ipost++) {
            lSpk = glbSpkDN[ipost];
            for (ipre = 0; ipre < 1000; ipre++) {
                scalar dt = t - (sTKC[ipre]) - ((1.00000000000000000e+01f)); 
                scalar dg =0; 
                if (dt > (3.12500000000000000e+01f))  
                    dg = -((7.49999999999999934e-05f)) ; 
                else if (dt > 0)  
                    dg = (-1.20000000000000003e-05f) * dt + ((2.99999999999999974e-04f)); 
                else if (dt > (-2.50124999999999993e+01f))  
                    dg = (1.20000000000000003e-05f) * dt + ((2.99999999999999974e-04f)); 
                else dg = -((1.49999999999999993e-07f)) ; 
                gRawKCDN[lSpk + 100 * ipre] += dg; 
                gKCDN[lSpk + 100 * ipre]=(1.49999999999999994e-02f)/2.0f *(tanhf((3.33299999999999983e+01f)*(gRawKCDN[lSpk + 100 * ipre] - ((7.49999999999999972e-03f))))+1); 
                
            }
        }
    }
}

#endif
