

#ifndef _HHVClamp_neuronFnct_cc
#define _HHVClamp_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model HHVClamp containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group HH
     {
        glbSpkCntHH[0] = 0;
        
        for (int n = 0; n < 12; n++) {
            scalar lV = VHH[n];
            scalar lm = mHH[n];
            scalar lh = hHH[n];
            scalar ln = nHH[n];
            scalar lgNa = gNaHH[n];
            scalar lENa = ENaHH[n];
            scalar lgK = gKHH[n];
            scalar lEK = EKHH[n];
            scalar lgl = glHH[n];
            scalar lEl = ElHH[n];
            scalar lC = CHH[n];
            scalar lerr = errHH[n];
            
            float Isyn = 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= (lV > 100);
            // calculate membrane potential
            scalar Imem;
            unsigned int mt;
            scalar mdt= DT/100.0f;
            scalar Icoupl;
            for (mt=0; mt < 100; mt++) {
               Icoupl= 200.0f*(stepVGHH-lV);
               Imem= -(lm*lm*lm*lh*lgNa*(lV-(lENa))+
                   ln*ln*ln*ln*lgK*(lV-(lEK))+
                   lgl*(lV-(lEl))-Icoupl);
               scalar _a= (3.5f+0.1f*lV) / (1.0f-expf(-3.5f-0.1f*lV));
               scalar _b= 4.0f*expf(-(lV+60.0f)/18.0f);
               lm+= (_a*(1.0f-lm)-_b*lm)*mdt;
               _a= 0.07f*expf(-lV/20.0f-3.0f);
               _b= 1.0f / (expf(-3.0f-0.1f*lV)+1.0f);
               lh+= (_a*(1.0f-lh)-_b*lh)*mdt;
               _a= (-0.5f-0.01f*lV) / (expf(-5.0f-0.1f*lV)-1.0f);
               _b= 0.125f*expf(-(lV+60.0f)/80.0f);
               ln+= (_a*(1.0f-ln)-_b*ln)*mdt;
               lV+= Imem/lC*mdt;
            }
            lerr+= abs(Icoupl-IsynGHH);
            
            // test for and register a true spike
            if ((lV > 100) && !(oldSpike)) {
                glbSpkHH[glbSpkCntHH[0]++] = n;
            }
            VHH[n] = lV;
            mHH[n] = lm;
            hHH[n] = lh;
            nHH[n] = ln;
            gNaHH[n] = lgNa;
            ENaHH[n] = lENa;
            gKHH[n] = lgK;
            EKHH[n] = lEK;
            glHH[n] = lgl;
            ElHH[n] = lEl;
            CHH[n] = lC;
            errHH[n] = lerr;
        }
    }
    
}

#endif
