

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
    unsigned int id = 128 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shSpk[128];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (threadIdx.x == 0) {
        spkCount = 0;
    }
    __syncthreads();
    
    // neuron group Excitatory
    if (id < 3200) {
        
        // only do this for existing neurons
        if (id < 3200) {
            // pull neuron variables in a coalesced access
            scalar lt_spike = dd_t_spikeExcitatory[id];
            scalar lv = dd_vExcitatory[id];
            unsigned int l_regimeID = dd__regimeIDExcitatory[id];
            
            float Isyn = 0;
            scalar I_syn = 0;
            // pull inSyn values in a coalesced access
            float linSynExcitatory_to_Excitatory_Synapse_0_weight_update = dd_inSynExcitatory_to_Excitatory_Synapse_0_weight_update[id];
            I_syn += linSynExcitatory_to_Excitatory_Synapse_0_weight_update*((0.00000000000000000e+00f)-lv);
            
            // pull inSyn values in a coalesced access
            float linSynInhibitory_to_Excitatory_Synapse_0_weight_update = dd_inSynInhibitory_to_Excitatory_Synapse_0_weight_update[id];
            I_syn += linSynInhibitory_to_Excitatory_Synapse_0_weight_update*((-8.00000000000000000e+01f)-lv);
            
            // pull inSyn values in a coalesced access
            float linSynSpike_Source_to_Excitatory_Synapse_0_weight_update = dd_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update[id];
            I_syn += linSynSpike_Source_to_Excitatory_Synapse_0_weight_update*((0.00000000000000000e+00f)-lv);
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if(l_regimeID == 0) {
                if(lv > (-5.00000000000000000e+01f)) {
                    lv = (-6.00000000000000000e+01f);
                    lt_spike = t;
                    l_regimeID = 1;
                }
                lv += DT * ((((0.00000000000000000e+00f) + I_syn) / (2.00000000000000011e-01f)) + ((-6.00000000000000000e+01f) - lv) / (2.00000000000000000e+01f));
            }
            else if(l_regimeID == 1) {
                if(t > (lt_spike + (5.00000000000000000e+00f))) {
                    l_regimeID = 0;
                }
            }
            
            // test for and register a true spike
            if ((l_regimeID == 0 && (lv > (-5.00000000000000000e+01f))))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
            }
            dd_t_spikeExcitatory[id] = lt_spike;
            dd_vExcitatory[id] = lv;
            dd__regimeIDExcitatory[id] = l_regimeID;
            // the post-synaptic dynamics
            linSynExcitatory_to_Excitatory_Synapse_0_weight_update *=  (9.80198673306755253e-01f);
            
            dd_inSynExcitatory_to_Excitatory_Synapse_0_weight_update[id] = linSynExcitatory_to_Excitatory_Synapse_0_weight_update;
            linSynInhibitory_to_Excitatory_Synapse_0_weight_update *=  (9.80198673306755253e-01f);
            
            dd_inSynInhibitory_to_Excitatory_Synapse_0_weight_update[id] = linSynInhibitory_to_Excitatory_Synapse_0_weight_update;
            linSynSpike_Source_to_Excitatory_Synapse_0_weight_update *=  (9.80198673306755253e-01f);
            
            dd_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update[id] = linSynSpike_Source_to_Excitatory_Synapse_0_weight_update;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntExcitatory[dd_spkQuePtrExcitatory], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkExcitatory[(dd_spkQuePtrExcitatory * 3200) + posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
    // neuron group Inhibitory
    if ((id >= 3200) && (id < 4096)) {
        unsigned int lid = id - 3200;
        
        // only do this for existing neurons
        if (lid < 800) {
            // pull neuron variables in a coalesced access
            scalar lt_spike = dd_t_spikeInhibitory[lid];
            scalar lv = dd_vInhibitory[lid];
            unsigned int l_regimeID = dd__regimeIDInhibitory[lid];
            
            float Isyn = 0;
            scalar I_syn = 0;
            // pull inSyn values in a coalesced access
            float linSynExcitatory_to_Inhibitory_Synapse_0_weight_update = dd_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update[lid];
            I_syn += linSynExcitatory_to_Inhibitory_Synapse_0_weight_update*((0.00000000000000000e+00f)-lv);
            
            // pull inSyn values in a coalesced access
            float linSynInhibitory_to_Inhibitory_Synapse_0_weight_update = dd_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update[lid];
            I_syn += linSynInhibitory_to_Inhibitory_Synapse_0_weight_update*((-8.00000000000000000e+01f)-lv);
            
            // pull inSyn values in a coalesced access
            float linSynSpike_Source_to_Inhibitory_Synapse_0_weight_update = dd_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update[lid];
            I_syn += linSynSpike_Source_to_Inhibitory_Synapse_0_weight_update*((0.00000000000000000e+00f)-lv);
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            if(l_regimeID == 0) {
                if(lv > (-5.00000000000000000e+01f)) {
                    lv = (-6.00000000000000000e+01f);
                    lt_spike = t;
                    l_regimeID = 1;
                }
                lv += DT * ((((0.00000000000000000e+00f) + I_syn) / (2.00000000000000011e-01f)) + ((-6.00000000000000000e+01f) - lv) / (2.00000000000000000e+01f));
            }
            else if(l_regimeID == 1) {
                if(t > (lt_spike + (5.00000000000000000e+00f))) {
                    l_regimeID = 0;
                }
            }
            
            // test for and register a true spike
            if ((l_regimeID == 0 && (lv > (-5.00000000000000000e+01f))))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
            }
            dd_t_spikeInhibitory[lid] = lt_spike;
            dd_vInhibitory[lid] = lv;
            dd__regimeIDInhibitory[lid] = l_regimeID;
            // the post-synaptic dynamics
            linSynExcitatory_to_Inhibitory_Synapse_0_weight_update *=  (9.80198673306755253e-01f);
            
            dd_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update[lid] = linSynExcitatory_to_Inhibitory_Synapse_0_weight_update;
            linSynInhibitory_to_Inhibitory_Synapse_0_weight_update *=  (9.80198673306755253e-01f);
            
            dd_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update[lid] = linSynInhibitory_to_Inhibitory_Synapse_0_weight_update;
            linSynSpike_Source_to_Inhibitory_Synapse_0_weight_update *=  (9.80198673306755253e-01f);
            
            dd_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update[lid] = linSynSpike_Source_to_Inhibitory_Synapse_0_weight_update;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntInhibitory[dd_spkQuePtrInhibitory], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkInhibitory[(dd_spkQuePtrInhibitory * 800) + posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
    // neuron group Spike_Source
    if ((id >= 4096) && (id < 4224)) {
        unsigned int lid = id - 4096;
        
        // only do this for existing neurons
        if (lid < 20) {
            // pull neuron variables in a coalesced access
            
            // test whether spike condition was fulfilled previously
            // calculate membrane potential
            
            // test for and register a true spike
            if (0)  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = lid;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntSpike_Source[0], spkCount);
        }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkSpike_Source[posSpk + threadIdx.x] = shSpk[threadIdx.x];
        }
    }
    
}

#endif
