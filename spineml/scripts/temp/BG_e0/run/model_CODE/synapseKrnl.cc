

#ifndef _model_synapseKrnl_cc
#define _model_synapseKrnl_cc
#define BLOCKSZ_SYN 32

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model model containing the synapse kernel and learning kernel functions.
*/
//-------------------------------------------------------------------------

#define BLOCKSZ_SYNDYN 32
extern "C" __global__ void calcSynapseDynamics(float t)
 {
    unsigned int id = BLOCKSZ_SYNDYN * blockIdx.x + threadIdx.x;
    float addtoinSyn;
    
    // execute internal synapse dynamics if any
    // synapse group Cortex_to_D1_Synapse_0_weight_update
    if (id < 32) {
        if (id < dd_indInGCortex_to_D1_Synapse_0_weight_update[6]) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (1.00000000000000000e+00f)*dd_outCortex[dd_preIndCortex_to_D1_Synapse_0_weight_update[id]];
            atomicAdd(&dd_inSynCortex_to_D1_Synapse_0_weight_update[dd_indCortex_to_D1_Synapse_0_weight_update[id]], addtoinSyn);
            
        }
    }
    // synapse group Cortex_to_D2_Synapse_0_weight_update
    if ((id >= 32) && (id < 64)) {
        unsigned int lid = id - 32;
        if (lid < dd_indInGCortex_to_D2_Synapse_0_weight_update[6]) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (1.00000000000000000e+00f)*dd_outCortex[dd_preIndCortex_to_D2_Synapse_0_weight_update[lid]];
            atomicAdd(&dd_inSynCortex_to_D2_Synapse_0_weight_update[dd_indCortex_to_D2_Synapse_0_weight_update[lid]], addtoinSyn);
            
        }
    }
    // synapse group Cortex_to_STN_Synapse_0_weight_update
    if ((id >= 64) && (id < 96)) {
        unsigned int lid = id - 64;
        if (lid < dd_indInGCortex_to_STN_Synapse_0_weight_update[6]) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (1.00000000000000000e+00f)*dd_outCortex[dd_preIndCortex_to_STN_Synapse_0_weight_update[lid]];
            atomicAdd(&dd_inSynCortex_to_STN_Synapse_0_weight_update[dd_indCortex_to_STN_Synapse_0_weight_update[lid]], addtoinSyn);
            
        }
    }
    // synapse group D1_to_GPi_Synapse_0_weight_update
    if ((id >= 96) && (id < 128)) {
        unsigned int lid = id - 96;
        if (lid < dd_indInGD1_to_GPi_Synapse_0_weight_update[6]) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (-1.00000000000000000e+00f)*dd_outD1[dd_preIndD1_to_GPi_Synapse_0_weight_update[lid]];
            atomicAdd(&dd_inSynD1_to_GPi_Synapse_0_weight_update[dd_indD1_to_GPi_Synapse_0_weight_update[lid]], addtoinSyn);
            
        }
    }
    // synapse group D2_to_GPe_Synapse_0_weight_update
    if ((id >= 128) && (id < 160)) {
        unsigned int lid = id - 128;
        if (lid < dd_indInGD2_to_GPe_Synapse_0_weight_update[6]) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (-1.00000000000000000e+00f)*dd_outD2[dd_preIndD2_to_GPe_Synapse_0_weight_update[lid]];
            atomicAdd(&dd_inSynD2_to_GPe_Synapse_0_weight_update[dd_indD2_to_GPe_Synapse_0_weight_update[lid]], addtoinSyn);
            
        }
    }
    // synapse group GPe_to_GPi_Synapse_0_weight_update
    if ((id >= 160) && (id < 192)) {
        unsigned int lid = id - 160;
        unsigned int delaySlot = (dd_spkQuePtrGPe + 2) % 2;
        if (lid < dd_indInGGPe_to_GPi_Synapse_0_weight_update[6]) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (-4.00000000000000022e-01f)*dd_outGPe[(delaySlot * 6) + dd_preIndGPe_to_GPi_Synapse_0_weight_update[lid]];
            atomicAdd(&dd_inSynGPe_to_GPi_Synapse_0_weight_update[dd_indGPe_to_GPi_Synapse_0_weight_update[lid]], addtoinSyn);
            
        }
    }
    // synapse group GPe_to_STN_Synapse_0_weight_update
    if ((id >= 192) && (id < 224)) {
        unsigned int lid = id - 192;
        unsigned int delaySlot = (dd_spkQuePtrGPe + 1) % 2;
        if (lid < dd_indInGGPe_to_STN_Synapse_0_weight_update[6]) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (-1.00000000000000000e+00f)*dd_outGPe[(delaySlot * 6) + dd_preIndGPe_to_STN_Synapse_0_weight_update[lid]];
            atomicAdd(&dd_inSynGPe_to_STN_Synapse_0_weight_update[dd_indGPe_to_STN_Synapse_0_weight_update[lid]], addtoinSyn);
            
        }
    }
    // synapse group STN_to_GPe_Synapse_0_weight_update
    if ((id >= 224) && (id < 288)) {
        unsigned int lid = id - 224;
        if (lid < 36) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (9.00000000000000022e-01f)*dd_outSTN[lid/6];
            atomicAdd(&dd_inSynSTN_to_GPe_Synapse_0_weight_update[lid%6], addtoinSyn);
            
        }
    }
    // synapse group STN_to_GPi_Synapse_0_weight_update
    if ((id >= 288) && (id < 352)) {
        unsigned int lid = id - 288;
        if (lid < 36) {
            // all threads participate that can work on an existing synapse
            addtoinSyn = (9.00000000000000022e-01f)*dd_outSTN[lid/6];
            atomicAdd(&dd_inSynSTN_to_GPi_Synapse_0_weight_update[lid%6], addtoinSyn);
            
        }
    }
}
extern "C" __global__ void calcSynapses(float t)
 {
    unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;
    unsigned int lmax, j, r;
    float addtoinSyn;
    volatile __shared__ float shLg[BLOCKSZ_SYN];
    float linSyn;
    unsigned int ipost;
    unsigned int prePos; 
    unsigned int npost; 
    
    // synapse group Cortex_to_D1_Synapse_0_weight_update
    if (id < 32) {
        // only do this for existing neurons
        if (id < 6) {
            linSyn = dd_inSynCortex_to_D1_Synapse_0_weight_update[id];
        }
        
        // only do this for existing neurons
        if (id < 6) {
            dd_inSynCortex_to_D1_Synapse_0_weight_update[id] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Cortex_to_D2_Synapse_0_weight_update
    if ((id >= 32) && (id < 64)) {
        unsigned int lid = id - 32;
        // only do this for existing neurons
        if (lid < 6) {
            linSyn = dd_inSynCortex_to_D2_Synapse_0_weight_update[lid];
        }
        
        // only do this for existing neurons
        if (lid < 6) {
            dd_inSynCortex_to_D2_Synapse_0_weight_update[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Cortex_to_STN_Synapse_0_weight_update
    if ((id >= 64) && (id < 96)) {
        unsigned int lid = id - 64;
        // only do this for existing neurons
        if (lid < 6) {
            linSyn = dd_inSynCortex_to_STN_Synapse_0_weight_update[lid];
        }
        
        // only do this for existing neurons
        if (lid < 6) {
            dd_inSynCortex_to_STN_Synapse_0_weight_update[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group D1_to_GPi_Synapse_0_weight_update
    if ((id >= 96) && (id < 128)) {
        unsigned int lid = id - 96;
        // only do this for existing neurons
        if (lid < 6) {
            linSyn = dd_inSynD1_to_GPi_Synapse_0_weight_update[lid];
        }
        
        // only do this for existing neurons
        if (lid < 6) {
            dd_inSynD1_to_GPi_Synapse_0_weight_update[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group D2_to_GPe_Synapse_0_weight_update
    if ((id >= 128) && (id < 160)) {
        unsigned int lid = id - 128;
        // only do this for existing neurons
        if (lid < 6) {
            linSyn = dd_inSynD2_to_GPe_Synapse_0_weight_update[lid];
        }
        
        // only do this for existing neurons
        if (lid < 6) {
            dd_inSynD2_to_GPe_Synapse_0_weight_update[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group GPe_to_GPi_Synapse_0_weight_update
    if ((id >= 160) && (id < 192)) {
        unsigned int lid = id - 160;
        unsigned int delaySlot = (dd_spkQuePtrGPe + 2) % 2;
        // only do this for existing neurons
        if (lid < 6) {
            linSyn = dd_inSynGPe_to_GPi_Synapse_0_weight_update[lid];
        }
        
        // only do this for existing neurons
        if (lid < 6) {
            dd_inSynGPe_to_GPi_Synapse_0_weight_update[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group GPe_to_STN_Synapse_0_weight_update
    if ((id >= 192) && (id < 224)) {
        unsigned int lid = id - 192;
        unsigned int delaySlot = (dd_spkQuePtrGPe + 1) % 2;
        // only do this for existing neurons
        if (lid < 6) {
            linSyn = dd_inSynGPe_to_STN_Synapse_0_weight_update[lid];
        }
        
        // only do this for existing neurons
        if (lid < 6) {
            dd_inSynGPe_to_STN_Synapse_0_weight_update[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group STN_to_GPe_Synapse_0_weight_update
    if ((id >= 224) && (id < 256)) {
        unsigned int lid = id - 224;
        // only do this for existing neurons
        if (lid < 6) {
            linSyn = dd_inSynSTN_to_GPe_Synapse_0_weight_update[lid];
        }
        
        // only do this for existing neurons
        if (lid < 6) {
            dd_inSynSTN_to_GPe_Synapse_0_weight_update[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group STN_to_GPi_Synapse_0_weight_update
    if ((id >= 256) && (id < 288)) {
        unsigned int lid = id - 256;
        // only do this for existing neurons
        if (lid < 6) {
            linSyn = dd_inSynSTN_to_GPi_Synapse_0_weight_update[lid];
        }
        
        // only do this for existing neurons
        if (lid < 6) {
            dd_inSynSTN_to_GPi_Synapse_0_weight_update[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 8) {
                dd_glbSpkCntCortex[0] = 0;
                dd_glbSpkCntD1[0] = 0;
                dd_glbSpkCntD2[0] = 0;
                dd_spkQuePtrGPe = (dd_spkQuePtrGPe + 1) % 2;
                dd_glbSpkCntGPe[0] = 0;
                dd_glbSpkCntGPi[0] = 0;
                dd_glbSpkCntSTN[0] = 0;
                d_done = 0;
            }
        }
    }
    
}


#endif
