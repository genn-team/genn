
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model model containing the host side code for a GPU simulator version.
*/
//-------------------------------------------------------------------------


// software version of atomic add for double precision
__device__ double atomicAddSW(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template<typename RNG>
__device__ float exponentialDistFloat(RNG *rng) {
    float a = 0.0f;
    while (true) {
        float u = curand_uniform(rng);
        const float u0 = u;
        while (true) {
            float uStar = curand_uniform(rng);
            if (u < uStar) {
                return  a + u0;
            }
            u = curand_uniform(rng);
            if (u >= uStar) {
                break;
            }
        }
        a += 1.0f;
    }
}

template<typename RNG>
__device__ double exponentialDistDouble(RNG *rng) {
    double a = 0.0f;
    while (true) {
        double u = curand_uniform_double(rng);
        const double u0 = u;
        while (true) {
            double uStar = curand_uniform_double(rng);
            if (u < uStar) {
                return  a + u0;
            }
            u = curand_uniform_double(rng);
            if (u >= uStar) {
                break;
            }
        }
        a += 1.0;
    }
}

#include "neuronKrnl.cc"
#include "synapseKrnl.cc"
// ------------------------------------------------------------------------
// copying things to device

void pushExcitatoryStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_t_spikeExcitatory, t_spikeExcitatory, 3200 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_vExcitatory, vExcitatory, 3200 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d__regimeIDExcitatory, _regimeIDExcitatory, 3200 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushExcitatorySpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntExcitatory, glbSpkCntExcitatory, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkExcitatory, glbSpkExcitatory, 6400 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushExcitatorySpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushExcitatoryCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntExcitatory+spkQuePtrExcitatory, glbSpkCntExcitatory+spkQuePtrExcitatory, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkExcitatory+(spkQuePtrExcitatory*3200), glbSpkExcitatory+(spkQuePtrExcitatory*3200), glbSpkCntExcitatory[spkQuePtrExcitatory] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushExcitatoryCurrentSpikeEventsToDevice()
 {
}

void pushInhibitoryStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_t_spikeInhibitory, t_spikeInhibitory, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_vInhibitory, vInhibitory, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d__regimeIDInhibitory, _regimeIDInhibitory, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushInhibitorySpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntInhibitory, glbSpkCntInhibitory, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkInhibitory, glbSpkInhibitory, 1600 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushInhibitorySpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushInhibitoryCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntInhibitory+spkQuePtrInhibitory, glbSpkCntInhibitory+spkQuePtrInhibitory, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkInhibitory+(spkQuePtrInhibitory*800), glbSpkInhibitory+(spkQuePtrInhibitory*800), glbSpkCntInhibitory[spkQuePtrInhibitory] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushInhibitoryCurrentSpikeEventsToDevice()
 {
}

void pushSpike_SourceStateToDevice(bool hostInitialisedOnly)
 {
}

void pushSpike_SourceSpikesToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntSpike_Source, glbSpkCntSpike_Source, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkSpike_Source, glbSpkSpike_Source, 20 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushSpike_SourceSpikeEventsToDevice(bool hostInitialisedOnly)
 {
}

void pushSpike_SourceCurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntSpike_Source, glbSpkCntSpike_Source, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkSpike_Source, glbSpkSpike_Source, glbSpkCntSpike_Source[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushSpike_SourceCurrentSpikeEventsToDevice()
 {
}

void pushExcitatory_to_Excitatory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynExcitatory_to_Excitatory_Synapse_0_weight_update, inSynExcitatory_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushExcitatory_to_Inhibitory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update, inSynExcitatory_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushInhibitory_to_Excitatory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynInhibitory_to_Excitatory_Synapse_0_weight_update, inSynInhibitory_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushInhibitory_to_Inhibitory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update, inSynInhibitory_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushSpike_Source_to_Excitatory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update, inSynSpike_Source_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaMemcpyHostToDevice));
}

void pushSpike_Source_to_Inhibitory_Synapse_0_weight_updateStateToDevice(bool hostInitialisedOnly)
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update, inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaMemcpyHostToDevice));
}

// ------------------------------------------------------------------------
// copying things from device

void pullExcitatoryStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_spikeExcitatory, d_t_spikeExcitatory, 3200 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(vExcitatory, d_vExcitatory, 3200 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(_regimeIDExcitatory, d__regimeIDExcitatory, 3200 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullExcitatorySpikeEventsFromDevice()
 {
}

void pullExcitatorySpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntExcitatory, d_glbSpkCntExcitatory, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkExcitatory, d_glbSpkExcitatory, glbSpkCntExcitatory [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullExcitatorySpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullExcitatoryCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntExcitatory+spkQuePtrExcitatory, d_glbSpkCntExcitatory+spkQuePtrExcitatory, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkExcitatory+(spkQuePtrExcitatory*3200), d_glbSpkExcitatory+(spkQuePtrExcitatory*3200), glbSpkCntExcitatory[spkQuePtrExcitatory] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullExcitatoryCurrentSpikeEventsFromDevice()
 {
}

void pullInhibitoryStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(t_spikeInhibitory, d_t_spikeInhibitory, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(vInhibitory, d_vInhibitory, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(_regimeIDInhibitory, d__regimeIDInhibitory, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullInhibitorySpikeEventsFromDevice()
 {
}

void pullInhibitorySpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInhibitory, d_glbSpkCntInhibitory, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkInhibitory, d_glbSpkInhibitory, glbSpkCntInhibitory [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullInhibitorySpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullInhibitoryCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInhibitory+spkQuePtrInhibitory, d_glbSpkCntInhibitory+spkQuePtrInhibitory, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkInhibitory+(spkQuePtrInhibitory*800), d_glbSpkInhibitory+(spkQuePtrInhibitory*800), glbSpkCntInhibitory[spkQuePtrInhibitory] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullInhibitoryCurrentSpikeEventsFromDevice()
 {
}

void pullSpike_SourceStateFromDevice()
 {
}

void pullSpike_SourceSpikeEventsFromDevice()
 {
}

void pullSpike_SourceSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntSpike_Source, d_glbSpkCntSpike_Source, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkSpike_Source, d_glbSpkSpike_Source, glbSpkCntSpike_Source [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullSpike_SourceSpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
}

void pullSpike_SourceCurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntSpike_Source, d_glbSpkCntSpike_Source, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkSpike_Source, d_glbSpkSpike_Source, glbSpkCntSpike_Source[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullSpike_SourceCurrentSpikeEventsFromDevice()
 {
}

void pullExcitatory_to_Excitatory_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynExcitatory_to_Excitatory_Synapse_0_weight_update, d_inSynExcitatory_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullExcitatory_to_Inhibitory_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynExcitatory_to_Inhibitory_Synapse_0_weight_update, d_inSynExcitatory_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullInhibitory_to_Excitatory_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynInhibitory_to_Excitatory_Synapse_0_weight_update, d_inSynInhibitory_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullInhibitory_to_Inhibitory_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynInhibitory_to_Inhibitory_Synapse_0_weight_update, d_inSynInhibitory_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullSpike_Source_to_Excitatory_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynSpike_Source_to_Excitatory_Synapse_0_weight_update, d_inSynSpike_Source_to_Excitatory_Synapse_0_weight_update, 3200 * sizeof(float), cudaMemcpyDeviceToHost));
}

void pullSpike_Source_to_Inhibitory_Synapse_0_weight_updateStateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update, d_inSynSpike_Source_to_Inhibitory_Synapse_0_weight_update, 800 * sizeof(float), cudaMemcpyDeviceToHost));
}

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice(bool hostInitialisedOnly)
 {
    pushExcitatoryStateToDevice(hostInitialisedOnly);
    pushExcitatorySpikesToDevice(hostInitialisedOnly);
    pushInhibitoryStateToDevice(hostInitialisedOnly);
    pushInhibitorySpikesToDevice(hostInitialisedOnly);
    pushSpike_SourceStateToDevice(hostInitialisedOnly);
    pushSpike_SourceSpikesToDevice(hostInitialisedOnly);
    pushExcitatory_to_Excitatory_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushExcitatory_to_Inhibitory_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushInhibitory_to_Excitatory_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushInhibitory_to_Inhibitory_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushSpike_Source_to_Excitatory_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
    pushSpike_Source_to_Inhibitory_Synapse_0_weight_updateStateToDevice(hostInitialisedOnly);
}

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushExcitatorySpikesToDevice();
    pushInhibitorySpikesToDevice();
    pushSpike_SourceSpikesToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushExcitatoryCurrentSpikesToDevice();
    pushInhibitoryCurrentSpikesToDevice();
    pushSpike_SourceCurrentSpikesToDevice();
}
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushExcitatorySpikeEventsToDevice();
    pushInhibitorySpikeEventsToDevice();
    pushSpike_SourceSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushExcitatoryCurrentSpikeEventsToDevice();
    pushInhibitoryCurrentSpikeEventsToDevice();
    pushSpike_SourceCurrentSpikeEventsToDevice();
}
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullExcitatoryStateFromDevice();
    pullExcitatorySpikesFromDevice();
    pullInhibitoryStateFromDevice();
    pullInhibitorySpikesFromDevice();
    pullSpike_SourceStateFromDevice();
    pullSpike_SourceSpikesFromDevice();
    pullExcitatory_to_Excitatory_Synapse_0_weight_updateStateFromDevice();
    pullExcitatory_to_Inhibitory_Synapse_0_weight_updateStateFromDevice();
    pullInhibitory_to_Excitatory_Synapse_0_weight_updateStateFromDevice();
    pullInhibitory_to_Inhibitory_Synapse_0_weight_updateStateFromDevice();
    pullSpike_Source_to_Excitatory_Synapse_0_weight_updateStateFromDevice();
    pullSpike_Source_to_Inhibitory_Synapse_0_weight_updateStateFromDevice();
}

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
    pullExcitatorySpikesFromDevice();
    pullInhibitorySpikesFromDevice();
    pullSpike_SourceSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
    pullExcitatoryCurrentSpikesFromDevice();
    pullInhibitoryCurrentSpikesFromDevice();
    pullSpike_SourceCurrentSpikesFromDevice();
}


// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntExcitatory, d_glbSpkCntExcitatory, 2* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntInhibitory, d_glbSpkCntInhibitory, 2* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntSpike_Source, d_glbSpkCntSpike_Source, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
}


// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
    pullExcitatorySpikeEventsFromDevice();
    pullInhibitorySpikeEventsFromDevice();
    pullSpike_SourceSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
    pullExcitatoryCurrentSpikeEventsFromDevice();
    pullInhibitoryCurrentSpikeEventsFromDevice();
    pullSpike_SourceCurrentSpikeEventsFromDevice();
}


// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void copySpikeEventNFromDevice()
 {
    
}


// ------------------------------------------------------------------------
// the time stepping procedure (using GPU)
void stepTimeGPU()
 {
    
    //model.padSumSynapseTrgN[model.synapseGrpN - 1] is 480
    dim3 sThreads(32, 1);
    dim3 sGrid(15, 1);
    
    dim3 nThreads(128, 1);
    dim3 nGrid(33, 1);
    
    calcSynapses <<< sGrid, sThreads >>> (t);
    spkQuePtrExcitatory = (spkQuePtrExcitatory + 1) % 2;
    spkQuePtrInhibitory = (spkQuePtrInhibitory + 1) % 2;
    calcNeurons <<< nGrid, nThreads >>> (t);
    iT++;
    t= iT*DT;
}

