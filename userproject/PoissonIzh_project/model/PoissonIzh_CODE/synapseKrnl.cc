

#ifndef _PoissonIzh_synapseKrnl_cc
#define _PoissonIzh_synapseKrnl_cc
#define BLOCKSZ_SYN 32

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model PoissonIzh containing the synapse kernel and learning kernel functions.
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
    
    // synapse group PNIzh1
    if (id < 32) {
        // only do this for existing neurons
        if (id < 10) {
            linSyn = dd_inSynPNIzh1[id];
        }
        lscnt = dd_glbSpkCntPN[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkPN[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 10) {
                    ipost = id;
                    addtoinSyn = dd_gPNIzh1[shSpk[j] * 10+ ipost];
                    linSyn += addtoinSyn;
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (id < 10) {
            dd_inSynPNIzh1[id] = linSyn;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 0) {
                dd_glbSpkCntIzh1[0] = 0;
                dd_glbSpkCntPN[0] = 0;
                d_done = 0;
            }
        }
    }
    
}


#endif
