

#ifndef _model_synapseKrnl_cc
#define _model_synapseKrnl_cc
#define BLOCKSZ_SYN 32

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model model containing the synapse kernel and learning kernel functions.
*/
//-------------------------------------------------------------------------

extern "C" __global__ void calcSynapses(float t)
 {
    unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;
    unsigned int lmax, j, r;
    float addtoinSyn;
    volatile __shared__ float shLg[BLOCKSZ_SYN];
    unsigned int ipost;
    unsigned int prePos; 
    unsigned int npost; 
    __shared__ unsigned int shSpk[BLOCKSZ_SYN];
    unsigned int lscnt, numSpikeSubsets;
    
    // synapse group Excitatory_to_Excitatory_Synapse_0_weight_update
    if (id < 128) {
        unsigned int delaySlot = (dd_spkQuePtrExcitatory + 1) % 2;
        lscnt = dd_glbSpkCntExcitatory[delaySlot];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkExcitatory[(delaySlot * 3200) + (r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 111) {
                    prePos = dd_indInGExcitatory_to_Excitatory_Synapse_0_weight_update[shSpk[j]];
                    npost = dd_indInGExcitatory_to_Excitatory_Synapse_0_weight_update[shSpk[j] + 1] - prePos;
                    if (id < npost) {
                        prePos += id;
                        ipost = dd_indExcitatory_to_Excitatory_Synapse_0_weight_update[prePos];
                        addtoinSyn = (4.00000000000000008e-03f);
                        atomicAdd(&dd_inSynExcitatory_to_Excitatory_Synapse_0_weight_update[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 14) {
                dd_spkQuePtrExcitatory = (dd_spkQuePtrExcitatory + 1) % 2;
                dd_glbSpkCntExcitatory[dd_spkQuePtrExcitatory] = 0;
                dd_spkQuePtrInhibitory = (dd_spkQuePtrInhibitory + 1) % 2;
                dd_glbSpkCntInhibitory[dd_spkQuePtrInhibitory] = 0;
                dd_glbSpkCntSpike_Source[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Excitatory_to_Inhibitory_Synapse_0_weight_update
    if ((id >= 128) && (id < 192)) {
        unsigned int lid = id - 128;
        unsigned int delaySlot = (dd_spkQuePtrExcitatory + 1) % 2;
        lscnt = dd_glbSpkCntExcitatory[delaySlot];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkExcitatory[(delaySlot * 3200) + (r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 41) {
                    prePos = dd_indInGExcitatory_to_Inhibitory_Synapse_0_weight_update[shSpk[j]];
                    npost = dd_indInGExcitatory_to_Inhibitory_Synapse_0_weight_update[shSpk[j] + 1] - prePos;
                    if (lid < npost) {
                        prePos += lid;
                        ipost = dd_indExcitatory_to_Inhibitory_Synapse_0_weight_update[prePos];
                        addtoinSyn = (4.00000000000000008e-03f);
                        atomicAdd(&dd_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 14) {
                dd_spkQuePtrExcitatory = (dd_spkQuePtrExcitatory + 1) % 2;
                dd_glbSpkCntExcitatory[dd_spkQuePtrExcitatory] = 0;
                dd_spkQuePtrInhibitory = (dd_spkQuePtrInhibitory + 1) % 2;
                dd_glbSpkCntInhibitory[dd_spkQuePtrInhibitory] = 0;
                dd_glbSpkCntSpike_Source[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Inhibitory_to_Excitatory_Synapse_0_weight_update
    if ((id >= 192) && (id < 320)) {
        unsigned int lid = id - 192;
        unsigned int delaySlot = (dd_spkQuePtrInhibitory + 1) % 2;
        lscnt = dd_glbSpkCntInhibitory[delaySlot];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkInhibitory[(delaySlot * 800) + (r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 109) {
                    prePos = dd_indInGInhibitory_to_Excitatory_Synapse_0_weight_update[shSpk[j]];
                    npost = dd_indInGInhibitory_to_Excitatory_Synapse_0_weight_update[shSpk[j] + 1] - prePos;
                    if (lid < npost) {
                        prePos += lid;
                        ipost = dd_indInhibitory_to_Excitatory_Synapse_0_weight_update[prePos];
                        addtoinSyn = (5.09999999999999967e-02f);
                        atomicAdd(&dd_inSynInhibitory_to_Excitatory_Synapse_0_weight_update[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 14) {
                dd_spkQuePtrExcitatory = (dd_spkQuePtrExcitatory + 1) % 2;
                dd_glbSpkCntExcitatory[dd_spkQuePtrExcitatory] = 0;
                dd_spkQuePtrInhibitory = (dd_spkQuePtrInhibitory + 1) % 2;
                dd_glbSpkCntInhibitory[dd_spkQuePtrInhibitory] = 0;
                dd_glbSpkCntSpike_Source[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Inhibitory_to_Inhibitory_Synapse_0_weight_update
    if ((id >= 320) && (id < 384)) {
        unsigned int lid = id - 320;
        unsigned int delaySlot = (dd_spkQuePtrInhibitory + 1) % 2;
        lscnt = dd_glbSpkCntInhibitory[delaySlot];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkInhibitory[(delaySlot * 800) + (r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 40) {
                    prePos = dd_indInGInhibitory_to_Inhibitory_Synapse_0_weight_update[shSpk[j]];
                    npost = dd_indInGInhibitory_to_Inhibitory_Synapse_0_weight_update[shSpk[j] + 1] - prePos;
                    if (lid < npost) {
                        prePos += lid;
                        ipost = dd_indInhibitory_to_Inhibitory_Synapse_0_weight_update[prePos];
                        addtoinSyn = (5.09999999999999967e-02f);
                        atomicAdd(&dd_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 14) {
                dd_spkQuePtrExcitatory = (dd_spkQuePtrExcitatory + 1) % 2;
                dd_glbSpkCntExcitatory[dd_spkQuePtrExcitatory] = 0;
                dd_spkQuePtrInhibitory = (dd_spkQuePtrInhibitory + 1) % 2;
                dd_glbSpkCntInhibitory[dd_spkQuePtrInhibitory] = 0;
                dd_glbSpkCntSpike_Source[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Spike_Source_to_Excitatory_Synapse_0_weight_update
    if ((id >= 384) && (id < 448)) {
        unsigned int lid = id - 384;
        lscnt = dd_glbSpkCntSpike_Source[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkSpike_Source[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 60) {
                    prePos = dd_indInGSpike_Source_to_Excitatory_Synapse_0_weight_update[shSpk[j]];
                    npost = dd_indInGSpike_Source_to_Excitatory_Synapse_0_weight_update[shSpk[j] + 1] - prePos;
                    if (lid < npost) {
                        prePos += lid;
                        ipost = dd_indSpike_Source_to_Excitatory_Synapse_0_weight_update[prePos];
                        addtoinSyn = (1.00000000000000006e-01f);
                        atomicAdd(&dd_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 14) {
                dd_spkQuePtrExcitatory = (dd_spkQuePtrExcitatory + 1) % 2;
                dd_glbSpkCntExcitatory[dd_spkQuePtrExcitatory] = 0;
                dd_spkQuePtrInhibitory = (dd_spkQuePtrInhibitory + 1) % 2;
                dd_glbSpkCntInhibitory[dd_spkQuePtrInhibitory] = 0;
                dd_glbSpkCntSpike_Source[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Spike_Source_to_Inhibitory_Synapse_0_weight_update
    if ((id >= 448) && (id < 480)) {
        unsigned int lid = id - 448;
        lscnt = dd_glbSpkCntSpike_Source[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkSpike_Source[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 23) {
                    prePos = dd_indInGSpike_Source_to_Inhibitory_Synapse_0_weight_update[shSpk[j]];
                    npost = dd_indInGSpike_Source_to_Inhibitory_Synapse_0_weight_update[shSpk[j] + 1] - prePos;
                    if (lid < npost) {
                        prePos += lid;
                        ipost = dd_indSpike_Source_to_Inhibitory_Synapse_0_weight_update[prePos];
                        addtoinSyn = (1.00000000000000006e-01f);
                        atomicAdd(&dd_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 14) {
                dd_spkQuePtrExcitatory = (dd_spkQuePtrExcitatory + 1) % 2;
                dd_glbSpkCntExcitatory[dd_spkQuePtrExcitatory] = 0;
                dd_spkQuePtrInhibitory = (dd_spkQuePtrInhibitory + 1) % 2;
                dd_glbSpkCntInhibitory[dd_spkQuePtrInhibitory] = 0;
                dd_glbSpkCntSpike_Source[0] = 0;
                d_done = 0;
            }
        }
    }
    
}


#endif
