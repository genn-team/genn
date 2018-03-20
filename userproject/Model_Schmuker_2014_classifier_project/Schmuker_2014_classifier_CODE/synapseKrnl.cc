

#ifndef _Schmuker_2014_classifier_synapseKrnl_cc
#define _Schmuker_2014_classifier_synapseKrnl_cc
#define BLOCKSZ_SYN 64

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model Schmuker_2014_classifier containing the synapse kernel and learning kernel functions.
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
    __shared__ unsigned int shSpkEvnt[BLOCKSZ_SYN];
    unsigned int lscntEvnt, numSpikeSubsetsEvnt;
    
    // synapse group ANAN
    if (id < 192) {
        // only do this for existing neurons
        if (id < 180) {
            linSyn = dd_inSynANAN[id];
            }
        lscntEvnt = dd_glbSpkCntEvntAN[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpkEvnt[threadIdx.x] = dd_glbSpkEvntAN[(r * BLOCKSZ_SYN) + threadIdx.x];
                }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 180) {
                    ipost = id;
                    addtoinSyn = dd_gANAN[shSpkEvnt[j] * 180+ ipost];
linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (id < 180) {
            dd_inSynANAN[id] = linSyn;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 25) {
                dd_glbSpkCntEvntAN[0] = 0;
                dd_glbSpkCntAN[0] = 0;
                dd_glbSpkCntEvntPN[0] = 0;
                dd_glbSpkCntPN[0] = 0;
                dd_glbSpkCntEvntRN[0] = 0;
                dd_glbSpkCntRN[0] = 0;
                d_done = 0;
                }
            }
        }
    
    // synapse group PNAN
    if ((id >= 192) && (id < 384)) {
        unsigned int lid = id - 192;
        // only do this for existing neurons
        if (lid < 180) {
            linSyn = dd_inSynPNAN[lid];
            }
        lscntEvnt = dd_glbSpkCntEvntPN[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpkEvnt[threadIdx.x] = dd_glbSpkEvntPN[(r * BLOCKSZ_SYN) + threadIdx.x];
                }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 180) {
                    if (dd_VPN[shSpkEvnt[j]] > (-2.00000000000000000e+01f)) {
                        ipost = lid;
                        addtoinSyn = dd_gPNAN[shSpkEvnt[j] * 180+ ipost];
linSyn += addtoinSyn;

                        }
                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 180) {
            dd_inSynPNAN[lid] = linSyn;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 25) {
                dd_glbSpkCntEvntAN[0] = 0;
                dd_glbSpkCntAN[0] = 0;
                dd_glbSpkCntEvntPN[0] = 0;
                dd_glbSpkCntPN[0] = 0;
                dd_glbSpkCntEvntRN[0] = 0;
                dd_glbSpkCntRN[0] = 0;
                d_done = 0;
                }
            }
        }
    
    // synapse group PNPN
    if ((id >= 384) && (id < 1024)) {
        unsigned int lid = id - 384;
        // only do this for existing neurons
        if (lid < 600) {
            linSyn = dd_inSynPNPN[lid];
            }
        lscntEvnt = dd_glbSpkCntEvntPN[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpkEvnt[threadIdx.x] = dd_glbSpkEvntPN[(r * BLOCKSZ_SYN) + threadIdx.x];
                }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 600) {
                    if (dd_VPN[shSpkEvnt[j]] > (-3.50000000000000000e+01f)) {
                        ipost = lid;
                        addtoinSyn = dd_gPNPN[shSpkEvnt[j] * 600+ ipost];
linSyn += addtoinSyn;

                        }
                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 600) {
            dd_inSynPNPN[lid] = linSyn;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 25) {
                dd_glbSpkCntEvntAN[0] = 0;
                dd_glbSpkCntAN[0] = 0;
                dd_glbSpkCntEvntPN[0] = 0;
                dd_glbSpkCntPN[0] = 0;
                dd_glbSpkCntEvntRN[0] = 0;
                dd_glbSpkCntRN[0] = 0;
                d_done = 0;
                }
            }
        }
    
    // synapse group RNPN
    if ((id >= 1024) && (id < 1664)) {
        unsigned int lid = id - 1024;
        // only do this for existing neurons
        if (lid < 600) {
            linSyn = dd_inSynRNPN[lid];
            }
        lscntEvnt = dd_glbSpkCntEvntRN[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpkEvnt[threadIdx.x] = dd_glbSpkEvntRN[(r * BLOCKSZ_SYN) + threadIdx.x];
                }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 600) {
                    ipost = lid;
                    addtoinSyn = dd_gRNPN[shSpkEvnt[j] * 600+ ipost];
linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 600) {
            dd_inSynRNPN[lid] = linSyn;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 25) {
                dd_glbSpkCntEvntAN[0] = 0;
                dd_glbSpkCntAN[0] = 0;
                dd_glbSpkCntEvntPN[0] = 0;
                dd_glbSpkCntPN[0] = 0;
                dd_glbSpkCntEvntRN[0] = 0;
                dd_glbSpkCntRN[0] = 0;
                d_done = 0;
                }
            }
        }
    
    }


#endif
