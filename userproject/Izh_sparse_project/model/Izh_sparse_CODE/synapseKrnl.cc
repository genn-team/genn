

#ifndef _Izh_sparse_synapseKrnl_cc
#define _Izh_sparse_synapseKrnl_cc
#define BLOCKSZ_SYN 96

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model Izh_sparse containing the synapse kernel and learning kernel functions.
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
    
    // synapse group Exc_Exc
    if (id < 864) {
        lscnt = dd_glbSpkCntPExc[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkPExc[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 844) {
                    prePos = dd_indInGExc_Exc[shSpk[j]];
                    npost = dd_indInGExc_Exc[shSpk[j] + 1] - prePos;
                    if (id < npost) {
                        prePos += id;
                        ipost = dd_indExc_Exc[prePos];
                        addtoinSyn = dd_gExc_Exc[prePos];
                        atomicAdd(&dd_inSynExc_Exc[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 24) {
                dd_glbSpkCntPExc[0] = 0;
                dd_glbSpkCntPInh[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Exc_Inh
    if ((id >= 864) && (id < 1248)) {
        unsigned int lid = id - 864;
        lscnt = dd_glbSpkCntPExc[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkPExc[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 294) {
                    prePos = dd_indInGExc_Inh[shSpk[j]];
                    npost = dd_indInGExc_Inh[shSpk[j] + 1] - prePos;
                    if (lid < npost) {
                        prePos += lid;
                        ipost = dd_indExc_Inh[prePos];
                        addtoinSyn = dd_gExc_Inh[prePos];
                        atomicAdd(&dd_inSynExc_Inh[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 24) {
                dd_glbSpkCntPExc[0] = 0;
                dd_glbSpkCntPInh[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Inh_Exc
    if ((id >= 1248) && (id < 2112)) {
        unsigned int lid = id - 1248;
        lscnt = dd_glbSpkCntPInh[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkPInh[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 841) {
                    prePos = dd_indInGInh_Exc[shSpk[j]];
                    npost = dd_indInGInh_Exc[shSpk[j] + 1] - prePos;
                    if (lid < npost) {
                        prePos += lid;
                        ipost = dd_indInh_Exc[prePos];
                        addtoinSyn = dd_gInh_Exc[prePos];
                        atomicAdd(&dd_inSynInh_Exc[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 24) {
                dd_glbSpkCntPExc[0] = 0;
                dd_glbSpkCntPInh[0] = 0;
                d_done = 0;
            }
        }
    }
    
    // synapse group Inh_Inh
    if ((id >= 2112) && (id < 2400)) {
        unsigned int lid = id - 2112;
        lscnt = dd_glbSpkCntPInh[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkPInh[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 245) {
                    prePos = dd_indInGInh_Inh[shSpk[j]];
                    npost = dd_indInGInh_Inh[shSpk[j] + 1] - prePos;
                    if (lid < npost) {
                        prePos += lid;
                        ipost = dd_indInh_Inh[prePos];
                        addtoinSyn = dd_gInh_Inh[prePos];
                        atomicAdd(&dd_inSynInh_Inh[ipost], addtoinSyn);
                        
                    }
                }
                
            }
            
        }
        
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 24) {
                dd_glbSpkCntPExc[0] = 0;
                dd_glbSpkCntPInh[0] = 0;
                d_done = 0;
            }
        }
    }
    
}


#endif
