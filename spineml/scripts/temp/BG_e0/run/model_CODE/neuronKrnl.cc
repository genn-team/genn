

#ifndef _model_neuronKrnl_cc
#define _model_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model model containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(float t)
 {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    
    __syncthreads();
    
    // neuron group Cortex
    if (id < 32) {
        
        // only do this for existing neurons
        if (id < 6) {
            // pull neuron variables in a coalesced access
            scalar la = dd_aCortex[id];
            scalar lin = dd_inCortex[id];
            scalar lout = dd_outCortex[id];
            
            // calculate membrane potential
            if(la < -(0.00000000000000000e+00f)) {
                la = la=-(0.00000000000000000e+00f);
            }
            la += DT * ((-la+lin)/(1.00000000000000006e-01f));
            lout = (1.00000000000000000e+00f)*la+(0.00000000000000000e+00f);
            
            dd_aCortex[id] = la;
            dd_inCortex[id] = lin;
            dd_outCortex[id] = lout;
        }
        __syncthreads();
    }
    
    // neuron group D1
    if ((id >= 32) && (id < 64)) {
        unsigned int lid = id - 32;
        
        // only do this for existing neurons
        if (lid < 6) {
            // pull neuron variables in a coalesced access
            scalar la = dd_aD1[lid];
            scalar lout = dd_outD1[lid];
            
            float Isyn = 0;
            scalar in = 0;
            // pull inSyn values in a coalesced access
            float linSynCortex_to_D1_Synapse_0_weight_update = dd_inSynCortex_to_D1_Synapse_0_weight_update[lid];
            in += linSynCortex_to_D1_Synapse_0_weight_update;linSynCortex_to_D1_Synapse_0_weight_update = 0;
            
            // calculate membrane potential
            if(la<-(-2.00000000000000011e-01f)) {
                la = -(-2.00000000000000011e-01f);
            }
            la += DT * ((-la+in*(1+(2.00000000000000011e-01f)))/(2.50000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(-2.00000000000000011e-01f);
            
            dd_aD1[lid] = la;
            dd_outD1[lid] = lout;
            // the post-synaptic dynamics
            
            dd_inSynCortex_to_D1_Synapse_0_weight_update[lid] = linSynCortex_to_D1_Synapse_0_weight_update;
        }
        __syncthreads();
    }
    
    // neuron group D2
    if ((id >= 64) && (id < 96)) {
        unsigned int lid = id - 64;
        
        // only do this for existing neurons
        if (lid < 6) {
            // pull neuron variables in a coalesced access
            scalar la = dd_aD2[lid];
            scalar lout = dd_outD2[lid];
            
            float Isyn = 0;
            scalar in = 0;
            // pull inSyn values in a coalesced access
            float linSynCortex_to_D2_Synapse_0_weight_update = dd_inSynCortex_to_D2_Synapse_0_weight_update[lid];
            in += linSynCortex_to_D2_Synapse_0_weight_update;linSynCortex_to_D2_Synapse_0_weight_update = 0;
            
            // calculate membrane potential
            if(la<-(-2.00000000000000011e-01f)) {
                la = -(-2.00000000000000011e-01f);
            }
            la += DT * ((-la+in*(1+(-2.00000000000000011e-01f)))/(2.50000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(-2.00000000000000011e-01f);
            
            dd_aD2[lid] = la;
            dd_outD2[lid] = lout;
            // the post-synaptic dynamics
            
            dd_inSynCortex_to_D2_Synapse_0_weight_update[lid] = linSynCortex_to_D2_Synapse_0_weight_update;
        }
        __syncthreads();
    }
    
    // neuron group GPe
    if ((id >= 96) && (id < 128)) {
        unsigned int lid = id - 96;
        unsigned int delaySlot = (dd_spkQuePtrGPe + 1) % 2;
        
        // only do this for existing neurons
        if (lid < 6) {
            // pull neuron variables in a coalesced access
            scalar la = dd_aGPe[lid];
            scalar lout = dd_outGPe[(delaySlot * 6) + lid];
            
            float Isyn = 0;
            scalar in = 0;
            // pull inSyn values in a coalesced access
            float linSynD2_to_GPe_Synapse_0_weight_update = dd_inSynD2_to_GPe_Synapse_0_weight_update[lid];
            in += linSynD2_to_GPe_Synapse_0_weight_update;linSynD2_to_GPe_Synapse_0_weight_update = 0;
            
            // pull inSyn values in a coalesced access
            float linSynSTN_to_GPe_Synapse_0_weight_update = dd_inSynSTN_to_GPe_Synapse_0_weight_update[lid];
            in += linSynSTN_to_GPe_Synapse_0_weight_update;linSynSTN_to_GPe_Synapse_0_weight_update = 0;
            
            // calculate membrane potential
            if(la < -(2.00000000000000011e-01f)) {
                la = la=-(2.00000000000000011e-01f);
            }
            la += DT * ((-la+in)/(2.50000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(2.00000000000000011e-01f);
            
            dd_aGPe[lid] = la;
            dd_outGPe[(dd_spkQuePtrGPe * 6) + lid] = lout;
            // the post-synaptic dynamics
            
            dd_inSynD2_to_GPe_Synapse_0_weight_update[lid] = linSynD2_to_GPe_Synapse_0_weight_update;
            
            dd_inSynSTN_to_GPe_Synapse_0_weight_update[lid] = linSynSTN_to_GPe_Synapse_0_weight_update;
        }
        __syncthreads();
    }
    
    // neuron group GPi
    if ((id >= 128) && (id < 160)) {
        unsigned int lid = id - 128;
        
        // only do this for existing neurons
        if (lid < 6) {
            // pull neuron variables in a coalesced access
            scalar la = dd_aGPi[lid];
            scalar lout = dd_outGPi[lid];
            
            float Isyn = 0;
            scalar in = 0;
            // pull inSyn values in a coalesced access
            float linSynD1_to_GPi_Synapse_0_weight_update = dd_inSynD1_to_GPi_Synapse_0_weight_update[lid];
            in += linSynD1_to_GPi_Synapse_0_weight_update;linSynD1_to_GPi_Synapse_0_weight_update = 0;
            
            // pull inSyn values in a coalesced access
            float linSynSTN_to_GPi_Synapse_0_weight_update = dd_inSynSTN_to_GPi_Synapse_0_weight_update[lid];
            in += linSynSTN_to_GPi_Synapse_0_weight_update;linSynSTN_to_GPi_Synapse_0_weight_update = 0;
            
            // pull inSyn values in a coalesced access
            float linSynGPe_to_GPi_Synapse_0_weight_update = dd_inSynGPe_to_GPi_Synapse_0_weight_update[lid];
            in += linSynGPe_to_GPi_Synapse_0_weight_update;linSynGPe_to_GPi_Synapse_0_weight_update = 0;
            
            // calculate membrane potential
            if(la < -(2.00000000000000011e-01f)) {
                la = la=-(2.00000000000000011e-01f);
            }
            la += DT * ((-la+in)/(2.00000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(2.00000000000000011e-01f);
            
            dd_aGPi[lid] = la;
            dd_outGPi[lid] = lout;
            // the post-synaptic dynamics
            
            dd_inSynD1_to_GPi_Synapse_0_weight_update[lid] = linSynD1_to_GPi_Synapse_0_weight_update;
            
            dd_inSynSTN_to_GPi_Synapse_0_weight_update[lid] = linSynSTN_to_GPi_Synapse_0_weight_update;
            
            dd_inSynGPe_to_GPi_Synapse_0_weight_update[lid] = linSynGPe_to_GPi_Synapse_0_weight_update;
        }
        __syncthreads();
    }
    
    // neuron group STN
    if ((id >= 160) && (id < 192)) {
        unsigned int lid = id - 160;
        
        // only do this for existing neurons
        if (lid < 6) {
            // pull neuron variables in a coalesced access
            scalar la = dd_aSTN[lid];
            scalar lout = dd_outSTN[lid];
            
            float Isyn = 0;
            scalar in = 0;
            // pull inSyn values in a coalesced access
            float linSynGPe_to_STN_Synapse_0_weight_update = dd_inSynGPe_to_STN_Synapse_0_weight_update[lid];
            in += linSynGPe_to_STN_Synapse_0_weight_update;linSynGPe_to_STN_Synapse_0_weight_update = 0;
            
            // pull inSyn values in a coalesced access
            float linSynCortex_to_STN_Synapse_0_weight_update = dd_inSynCortex_to_STN_Synapse_0_weight_update[lid];
            in += linSynCortex_to_STN_Synapse_0_weight_update;linSynCortex_to_STN_Synapse_0_weight_update = 0;
            
            // calculate membrane potential
            if(la < -(2.50000000000000000e-01f)) {
                la = la=-(2.50000000000000000e-01f);
            }
            la += DT * ((-la+in)/(2.50000000000000000e+01f));
            lout = (1.00000000000000000e+00f)*la+(2.50000000000000000e-01f);
            
            dd_aSTN[lid] = la;
            dd_outSTN[lid] = lout;
            // the post-synaptic dynamics
            
            dd_inSynGPe_to_STN_Synapse_0_weight_update[lid] = linSynGPe_to_STN_Synapse_0_weight_update;
            
            dd_inSynCortex_to_STN_Synapse_0_weight_update[lid] = linSynCortex_to_STN_Synapse_0_weight_update;
        }
        __syncthreads();
    }
    
}

#endif
