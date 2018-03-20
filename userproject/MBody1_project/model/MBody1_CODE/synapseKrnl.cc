

#ifndef _MBody1_synapseKrnl_cc
#define _MBody1_synapseKrnl_cc
#define BLOCKSZ_SYN 96

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model MBody1 containing the synapse kernel and learning kernel functions.
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
    __shared__ unsigned int shSpkEvnt[BLOCKSZ_SYN];
    unsigned int lscntEvnt, numSpikeSubsetsEvnt;
    
    // synapse group DNDN
    if (id < 192) {
        // only do this for existing neurons
        if (id < 100) {
            linSyn = dd_inSynDNDN[id];
        }
        lscntEvnt = dd_glbSpkCntEvntDN[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpkEvnt[threadIdx.x] = dd_glbSpkEvntDN[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 100) {
                    ipost = id;
                    addtoinSyn = (5.00000000000000028e-02f) * tanhf((dd_VDN[shSpkEvnt[j]] - (-3.00000000000000000e+01f)) / (5.00000000000000000e+01f))* DT;
                    if (addtoinSyn < 0) addtoinSyn = 0.0f;
                    linSyn += addtoinSyn;
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (id < 100) {
            dd_inSynDNDN[id] = linSyn;
        }
    }
    
    // synapse group KCDN
    if ((id >= 192) && (id < 384)) {
        unsigned int lid = id - 192;
        // only do this for existing neurons
        if (lid < 100) {
            linSyn = dd_inSynKCDN[lid];
        }
        lscnt = dd_glbSpkCntKC[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkKC[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 100) {
                    ipost = lid;
                    addtoinSyn = dd_gKCDN[shSpk[j] * 100+ ipost];linSyn += addtoinSyn; 
                    scalar dt = dd_sTDN[ipost] - t - ((1.00000000000000000e+01f)); 
                    scalar dg = 0;
                    if (dt > (3.12500000000000000e+01f))  
                        dg = -((7.49999999999999934e-05f)) ; 
                    else if (dt > 0)  
                        dg = (-1.20000000000000003e-05f) * dt + ((2.99999999999999974e-04f)); 
                    else if (dt > (-2.50124999999999993e+01f))  
                        dg = (1.20000000000000003e-05f) * dt + ((2.99999999999999974e-04f)); 
                    else dg = - ((1.49999999999999993e-07f)) ; 
                    dd_gRawKCDN[shSpk[j] * 100+ ipost] += dg; 
                    dd_gKCDN[shSpk[j] * 100+ ipost]=(1.49999999999999994e-02f)/2 *(tanhf((3.33299999999999983e+01f)*(dd_gRawKCDN[shSpk[j] * 100+ ipost] - ((7.49999999999999972e-03f))))+1); 
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (lid < 100) {
            dd_inSynKCDN[lid] = linSyn;
        }
    }
    
    // synapse group LHIKC
    if ((id >= 384) && (id < 1440)) {
        unsigned int lid = id - 384;
        // only do this for existing neurons
        if (lid < 1000) {
            linSyn = dd_inSynLHIKC[lid];
        }
        lscntEvnt = dd_glbSpkCntEvntLHI[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpkEvnt[threadIdx.x] = dd_glbSpkEvntLHI[(r * BLOCKSZ_SYN) + threadIdx.x];
            }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 1000) {
                    ipost = lid;
                    addtoinSyn = (5.00000000000000028e-02f) * tanhf((dd_VLHI[shSpkEvnt[j]] - (-4.00000000000000000e+01f)) / (5.00000000000000000e+01f))* DT;
                    if (addtoinSyn < 0) addtoinSyn = 0.0f;
                    linSyn += addtoinSyn;
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (lid < 1000) {
            dd_inSynLHIKC[lid] = linSyn;
        }
    }
    
    // synapse group PNKC
    if ((id >= 1440) && (id < 2496)) {
        unsigned int lid = id - 1440;
        // only do this for existing neurons
        if (lid < 1000) {
            linSyn = dd_inSynPNKC[lid];
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
                if (lid < 1000) {
                    ipost = lid;
                    addtoinSyn = dd_gPNKC[shSpk[j] * 1000+ ipost];
                    linSyn += addtoinSyn;
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (lid < 1000) {
            dd_inSynPNKC[lid] = linSyn;
        }
    }
    
    // synapse group PNLHI
    if ((id >= 2496) && (id < 2592)) {
        unsigned int lid = id - 2496;
        // only do this for existing neurons
        if (lid < 20) {
            linSyn = dd_inSynPNLHI[lid];
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
                if (lid < 20) {
                    ipost = lid;
                    addtoinSyn = dd_gPNLHI[shSpk[j] * 20+ ipost];
                    linSyn += addtoinSyn;
                    
                }
                
            }
            
        }
        
        
        // only do this for existing neurons
        if (lid < 20) {
            dd_inSynPNLHI[lid] = linSyn;
        }
    }
    
}

extern "C" __global__ void learnSynapsesPost(float t)
 {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shSpk[32];
    unsigned int lscnt, numSpikeSubsets, lmax, j, r;
    
    // synapse group KCDN
    if (id < 1024) {
        lscnt = dd_glbSpkCntDN[0];
        numSpikeSubsets = (lscnt+31) / 32;
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % 32)+1;
            else lmax = 32;
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkDN[(r * 32) + threadIdx.x];
            }
            __syncthreads();
            // only work on existing neurons
            if (id < 1000) {
                // loop through all incoming spikes for learning
                for (j = 0; j < lmax; j++) {
                    
                    scalar dt = t - (dd_sTKC[id]) - ((1.00000000000000000e+01f)); 
                    scalar dg =0; 
                    if (dt > (3.12500000000000000e+01f))  
                        dg = -((7.49999999999999934e-05f)) ; 
                    else if (dt > 0)  
                        dg = (-1.20000000000000003e-05f) * dt + ((2.99999999999999974e-04f)); 
                    else if (dt > (-2.50124999999999993e+01f))  
                        dg = (1.20000000000000003e-05f) * dt + ((2.99999999999999974e-04f)); 
                    else dg = -((1.49999999999999993e-07f)) ; 
                    dd_gRawKCDN[id * 100 + shSpk[j]] += dg; 
                    dd_gKCDN[id * 100 + shSpk[j]]=(1.49999999999999994e-02f)/2.0f *(tanhf((3.33299999999999983e+01f)*(dd_gRawKCDN[id * 100 + shSpk[j]] - ((7.49999999999999972e-03f))))+1); 
                    
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 31) {
                dd_glbSpkCntEvntDN[0] = 0;
                dd_glbSpkCntDN[0] = 0;
                dd_glbSpkCntKC[0] = 0;
                dd_glbSpkCntEvntLHI[0] = 0;
                dd_glbSpkCntLHI[0] = 0;
                dd_glbSpkCntPN[0] = 0;
                d_done = 0;
            }
        }
    }
}

#endif
