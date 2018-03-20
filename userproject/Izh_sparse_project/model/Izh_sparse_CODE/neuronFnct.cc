

#ifndef _Izh_sparse_neuronFnct_cc
#define _Izh_sparse_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model Izh_sparse containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group PExc
     {
        glbSpkCntPExc[0] = 0;
        
        for (int n = 0; n < 8000; n++) {
            scalar lV = VPExc[n];
            scalar lU = UPExc[n];
            scalar la = aPExc[n];
            scalar lb = bPExc[n];
            scalar lc = cPExc[n];
            scalar ld = dPExc[n];
            
            float Isyn = 0;
            Isyn += inSynExc_Exc[n]; inSynExc_Exc[n] = 0;
            Isyn += inSynInh_Exc[n]; inSynInh_Exc[n] = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=lc;
               lU+=ld;
            } 
            const scalar i0 = (0.00000000000000000e+00f) + (standardNormalDistribution(rng) * (5.00000000000000000e+00f));
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+i0+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+i0+Isyn)*DT;
            lU+=la*(lb*lV-lU)*DT;
            //if (lV > 30.0f){      //keep this only for visualisation -- not really necessaary otherwise 
            //  lV=30.0f; 
            //}
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike)) {
                glbSpkPExc[glbSpkCntPExc[0]++] = n;
            }
            VPExc[n] = lV;
            UPExc[n] = lU;
            aPExc[n] = la;
            bPExc[n] = lb;
            cPExc[n] = lc;
            dPExc[n] = ld;
            // the post-synaptic dynamics
            
            // the post-synaptic dynamics
            
        }
    }
    
    // neuron group PInh
     {
        glbSpkCntPInh[0] = 0;
        
        for (int n = 0; n < 2000; n++) {
            scalar lV = VPInh[n];
            scalar lU = UPInh[n];
            scalar la = aPInh[n];
            scalar lb = bPInh[n];
            scalar lc = cPInh[n];
            scalar ld = dPInh[n];
            
            float Isyn = 0;
            Isyn += inSynExc_Inh[n]; inSynExc_Inh[n] = 0;
            Isyn += inSynInh_Inh[n]; inSynInh_Inh[n] = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV >= 29.99f);
            // calculate membrane potential
            if (lV >= 30.0f){
               lV=lc;
               lU+=ld;
            } 
            const scalar i0 = (0.00000000000000000e+00f) + (standardNormalDistribution(rng) * (2.00000000000000000e+00f));
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+i0+Isyn)*DT; //at two times for numerical stability
            lV+=0.5f*(0.04f*lV*lV+5.0f*lV+140.0f-lU+i0+Isyn)*DT;
            lU+=la*(lb*lV-lU)*DT;
            //if (lV > 30.0f){      //keep this only for visualisation -- not really necessaary otherwise 
            //  lV=30.0f; 
            //}
            
            // test for and register a true spike
            if ((lV >= 29.99f) && !(oldSpike)) {
                glbSpkPInh[glbSpkCntPInh[0]++] = n;
            }
            VPInh[n] = lV;
            UPInh[n] = lU;
            aPInh[n] = la;
            bPInh[n] = lb;
            cPInh[n] = lc;
            dPInh[n] = ld;
            // the post-synaptic dynamics
            
            // the post-synaptic dynamics
            
        }
    }
    
}

#endif
