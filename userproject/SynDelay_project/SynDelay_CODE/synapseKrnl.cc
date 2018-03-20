

#ifndef _SynDelay_synapseKrnl_cc
#define _SynDelay_synapseKrnl_cc
#define BLOCKSZ_SYN 64

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model SynDelay containing the synapse kernel and learning kernel functions.
*/
//-------------------------------------------------------------------------

extern "C" __global__ void calcSynapses(float t)
 {
    unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;
    unsigned int lmax, j, r;
    float addtoinSyn;
    volatile __shared__ float shLg[BLOCKSZ_SYN];
    float linSyn;
    unsigned int ipost;
    __shared__ unsigned int shSpk[BLOCKSZ_SYN];
    unsigned int lscnt, numSpikeSubsets;
    
    // synapse group InputInter
    if (id < 512) {
        unsigned int delaySlot = (dd_spkQuePtrInput + 4) % 7;
        // only do this for existing neurons
        if (id < 500) {
            linSyn = dd_inSynInputInter[id];
        }
        lscnt = dd_glbSpkCntInput[delaySlot];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkInput[(delaySlot * 500) + (r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 500) {
                    ipost = id;
                    addtoinSyn = (5.99999999999999978e-02f);
                    linSyn += addtoinSyn;
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (id < 500) {
            dd_inSynInputInter[id] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 23) {
                dd_spkQuePtrInput = (dd_spkQuePtrInput + 1) % 7;
                dd_glbSpkCntInput[dd_spkQuePtrInput] = 0;
                dd_glbSpkCntInter[0] = 0;
                dd_glbSpkCntOutput[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group InputOutput
    if ((id >= 512) && (id < 1024)) {
        unsigned int lid = id - 512;
        unsigned int delaySlot = (dd_spkQuePtrInput + 1) % 7;
        // only do this for existing neurons
        if (lid < 500) {
            linSyn = dd_inSynInputOutput[lid];
        }
        lscnt = dd_glbSpkCntInput[delaySlot];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkInput[(delaySlot * 500) + (r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 500) {
                    ipost = lid;
                    addtoinSyn = (2.99999999999999989e-02f);
                    linSyn += addtoinSyn;
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (lid < 500) {
            dd_inSynInputOutput[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 23) {
                dd_spkQuePtrInput = (dd_spkQuePtrInput + 1) % 7;
                dd_glbSpkCntInput[dd_spkQuePtrInput] = 0;
                dd_glbSpkCntInter[0] = 0;
                dd_glbSpkCntOutput[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group InterOutput
    if ((id >= 1024) && (id < 1536)) {
        unsigned int lid = id - 1024;
        // only do this for existing neurons
        if (lid < 500) {
            linSyn = dd_inSynInterOutput[lid];
        }
        lscnt = dd_glbSpkCntInter[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkInter[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 500) {
                    ipost = lid;
                    addtoinSyn = (2.99999999999999989e-02f);
                    linSyn += addtoinSyn;
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (lid < 500) {
            dd_inSynInterOutput[lid] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 23) {
                dd_spkQuePtrInput = (dd_spkQuePtrInput + 1) % 7;
                dd_glbSpkCntInput[dd_spkQuePtrInput] = 0;
                dd_glbSpkCntInter[0] = 0;
                dd_glbSpkCntOutput[0] = 0;
                d_done = 0;
            }
        }
    }
    
}


#endif
