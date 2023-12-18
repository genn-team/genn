#include "code_generator/backendBase.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "gennUtils.h"
#include "logging.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/groupMerged.h"
#include "code_generator/customConnectivityUpdateGroupMerged.h"
#include "code_generator/customUpdateGroupMerged.h"
#include "code_generator/initGroupMerged.h"
#include "code_generator/neuronUpdateGroupMerged.h"
#include "code_generator/synapseUpdateGroupMerged.h"

// GeNN runtime includes
#include "runtime/runtime.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
template<typename G>
void buildCustomUpdateSizeEnvironment(EnvironmentGroupMergedField<G> &env)
{
    // Add size field
    env.addField(Type::Uint32.addConst(), "size",
                Type::Uint32, "size", 
                [](const auto &, const auto &c, size_t) { return c.getSize(); });
}
//--------------------------------------------------------------------------
template<typename G>
void buildCustomUpdateWUSizeEnvironment(const BackendBase &backend, EnvironmentGroupMergedField<G> &env)
{
    // Synapse group fields 
    env.addField(Type::Uint32.addConst(), "num_pre",
                 Type::Uint32, "numSrcNeurons", 
                 [](const auto&, auto  &cg, size_t) { return cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); });
    env.addField(Type::Uint32.addConst(), "num_post",
                 Type::Uint32, "numTrgNeurons", 
                 [](const auto&, const auto  &cg, size_t) { return cg.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(); });
    env.addField(Type::Uint32, "_row_stride", "rowStride", 
                 [&backend](const auto&, const auto &cg, size_t) { return backend.getSynapticMatrixRowStride(*cg.getSynapseGroup()); });

    // If underlying synapse group has kernel connectivity
    const auto *sg = env.getGroup().getArchetype().getSynapseGroup();
    if(sg->getMatrixType() & SynapseMatrixWeight::KERNEL) {
        // Loop through kernel size dimensions
        // **TODO** automatic heterogeneity detection on all fields would make this much nicer
        std::ostringstream kernSizeInit;
        kernSizeInit << "const unsigned int kernelSize = ";
        const auto &kernelSize = env.getGroup().getArchetype().getKernelSize();
        for (size_t d = 0; d < kernelSize.size(); d++) {
            // If this dimension has a heterogeneous size, add it to struct
            if (isKernelSizeHeterogeneous(env.getGroup(), d)) {
                env.addField(Type::Uint32.addConst(), "_kernel_size_" + std::to_string(d), 
                             Type::Uint32, "kernelSize" + std::to_string(d),
                             [d](const auto&, const auto &g, size_t) { return g.getSynapseGroup()->getKernelSize().at(d); });
            }

            // Multiply size by dimension
            kernSizeInit << getKernelSize(env.getGroup(), d);
            if (d != (kernelSize.size() - 1)) {
                kernSizeInit << " * ";
            }
        }

        // Add size field
        kernSizeInit << ";";
        env.add(Type::Uint32.addConst(), "_kernel_size", "kernelSize",
                {env.addInitialiser(kernSizeInit.str())});
        env.add(Type::Uint32.addConst(), "_size", "$(_kernel_size)");
    }
    // Otherwise, calculate size as normal
    else {
        // Connectivity fields
        if(sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            env.addField(Type::Uint32.createPointer(), "_row_length", "rowLength",
                         [](const auto &runtime, const auto &cg, size_t) { return runtime.getArray(*cg.getSynapseGroup(), "rowLength"); });
            env.addField(sg->getSparseIndType().createPointer(), "_ind", "ind",
                         [](const auto &runtime, const auto &cg, size_t) { return runtime.getArray(*cg.getSynapseGroup(), "ind"); });
        }

        const auto indexType = backend.getSynapseIndexType(env.getGroup());
        const auto indexTypeName = indexType.getName();
        env.add(indexType.addConst(), "_size", "size",
                {env.addInitialiser("const " + indexTypeName + " size = (" + indexTypeName + ")$(num_pre) * $(_row_stride);")});
    }

}
//--------------------------------------------------------------------------
template<typename G>
void buildStandardNeuronEnvironment(EnvironmentGroupMergedField<G> &env, unsigned int batchSize)
{
    using namespace Type;

    env.addField(Uint32.addConst(), "num_neurons",
                 Uint32, "numNeurons",
                 [](const auto&, const auto &ng, size_t) { return ng.getNumNeurons(); });
    env.addField(Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "spkQuePtr"); });
    env.addField(Uint32.createPointer(), "_record_spk", "recordSpk",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g, "recordSpk"); });

    // If batching is enabled, calculate batch offset
    if(batchSize > 1) {
        env.add(Uint32.addConst(), "_batch_offset", "batchOffset",
                {env.addInitialiser("const unsigned int batchOffset = $(num_neurons) * $(batch);")});
    }
            
    // If axonal delays are required
    if(env.getGroup().getArchetype().isDelayRequired()) {
        // We should READ from delay slot before spkQuePtr
        const unsigned int numDelaySlots = env.getGroup().getArchetype().getNumDelaySlots();
        const std::string numDelaySlotsStr = std::to_string(numDelaySlots);
        env.add(Uint32.addConst(), "_read_delay_slot", "readDelaySlot",
                {env.addInitialiser("const unsigned int readDelaySlot = (*$(_spk_que_ptr) + " + std::to_string(numDelaySlots - 1) + ") % " + numDelaySlotsStr+ ";")});
        env.add(Uint32.addConst(), "_read_delay_offset", "readDelayOffset",
                {env.addInitialiser("const unsigned int readDelayOffset = $(_read_delay_slot) * $(num_neurons);")});

        // And we should WRITE to delay slot pointed to be spkQuePtr
        env.add(Uint32.addConst(), "_write_delay_slot", "writeDelaySlot",
                {env.addInitialiser("const unsigned int writeDelaySlot = *$(_spk_que_ptr);")});
        env.add(Uint32.addConst(), "_write_delay_offset", "writeDelayOffset",
                {env.addInitialiser("const unsigned int writeDelayOffset = $(_write_delay_slot) * $(num_neurons);")});

        // If batching is also enabled
        if(batchSize > 1) {
            // Calculate batched delay slots
            env.add(Uint32.addConst(), "_read_batch_delay_slot", "readBatchDelaySlot",
                    {env.addInitialiser("const unsigned int readBatchDelaySlot = ($(batch) * " + numDelaySlotsStr + ") + $(_read_delay_slot);")});
            env.add(Uint32.addConst(), "_write_batch_delay_slot", "writeBatchDelaySlot",
                    {env.addInitialiser("const unsigned int writeBatchDelaySlot = ($(batch) * " + numDelaySlotsStr + ") + $(_write_delay_slot);")});

            // Calculate current batch offset
            env.add(Uint32.addConst(), "_batch_delay_offset", "batchDelayOffset",
                    {env.addInitialiser("const unsigned int batchDelayOffset = $(_batch_offset) * " + numDelaySlotsStr + ";")});

            // Calculate further offsets to include delay and batch
            env.add(Uint32.addConst(), "_read_batch_delay_offset", "readBatchDelayOffset",
                    {env.addInitialiser("const unsigned int readBatchDelayOffset = $(_read_delay_offset) + $(_batch_delay_offset);")});
            env.add(Uint32.addConst(), "_write_batch_delay_offset", "writeBatchDelayOffset",
                    {env.addInitialiser("const unsigned int writeBatchDelayOffset = $(_write_delay_offset) + $(_batch_delay_offset);")});
        }
    }
}
//--------------------------------------------------------------------------
template<typename G>
void buildStandardSynapseEnvironment(const BackendBase &backend, EnvironmentGroupMergedField<G> &env, unsigned int batchSize)
{
    using namespace Type;

    // Synapse group fields 
    env.addField(Uint32.addConst(), "num_pre",
                 Uint32, "numSrcNeurons", 
                 [](const auto&, const SynapseGroupInternal &sg, size_t) { return sg.getSrcNeuronGroup()->getNumNeurons(); });
    env.addField(Uint32.addConst(), "num_post",
                 Uint32, "numTrgNeurons", 
                 [](const auto&, const SynapseGroupInternal &sg, size_t) { return sg.getTrgNeuronGroup()->getNumNeurons(); });
    env.addField(Uint32, "_row_stride", "rowStride", 
                 [&backend](const auto&, const SynapseGroupInternal &sg, size_t) { return backend.getSynapticMatrixRowStride(sg); });
    env.addField(Uint32, "_col_stride", "colStride", 
                 [](const auto&, const SynapseGroupInternal &sg, size_t) { return sg.getMaxSourceConnections(); });
                        
    // Postsynaptic model fields         
    env.addField(env.getGroup().getScalarType().createPointer(), "_out_post", "outPost",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g.getFusedPSTarget(), "outPost"); });
    env.addField(env.getGroup().getScalarType().createPointer(), "_den_delay", "denDelay",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g.getFusedPSTarget(), "denDelay"); });
    env.addField(Uint32.createPointer(), "_den_delay_ptr", "denDelayPtr",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g.getFusedPSTarget(), "denDelayPtr"); });
                       
    // Presynaptic output fields
    env.addField(env.getGroup().getScalarType().createPointer(), "_out_pre", "outPre",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(g.getFusedPreOutputTarget(), "outPre"); });
                        
    // Source neuron fields
    env.addField(Uint32.createPointer(), "_src_spk_que_ptr", "srcSpkQuePtr",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(*g.getSrcNeuronGroup(), "spkQuePtr"); });
    env.addField(Uint32.createPointer(), "_src_spk_cnt", "srcSpkCnt",
                 [](const auto &runtime, const auto &g, size_t){ return runtime.getFusedSrcSpikeArray(g, "SpkCnt"); });
    env.addField(Uint32.createPointer(), "_src_spk", "srcSpk",
                 [](const auto &runtime, const auto &g, size_t){ return runtime.getFusedSrcSpikeArray(g, "Spk"); });
    env.addField(Uint32.createPointer(), "_src_spk_cnt_event", "srcSpkCntEvent",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedSrcSpikeEventArray(g, "SpkCntEvent"); });
    env.addField(Uint32.createPointer(), "_src_spk_event", "srcSpkEvent",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedSrcSpikeEventArray(g, "SpkEvent"); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_src_st", "srcST",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedSrcSpikeArray(g, "ST"); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_src_prev_st", "srcPrevST",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedSrcSpikeArray(g, "PrevST"); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_src_set", "srcSET",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedSrcSpikeEventArray(g, "SET"); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_src_prev_set", "srcPrevSET",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedSrcSpikeEventArray(g, "PrevSET" ); });
    
    // Target neuron fields
    env.addField(Uint32.createPointer(), "_trg_spk_que_ptr", "trgSpkQuePtr",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(*g.getTrgNeuronGroup(), "spkQuePtr"); });
    env.addField(Uint32.createPointer(), "_trg_spk_cnt", "trgSpkCnt",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedTrgSpikeArray(g, "SpkCnt"); });
    env.addField(Uint32.createPointer(), "_trg_spk", "trgSpk",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedTrgSpikeArray(g, "Spk"); });
    env.addField(Uint32.createPointer(), "_trg_spk_cnt_event", "trgSpkCntEvent",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedTrgSpikeEventArray(g, "SpkCntEvent"); });
    env.addField(Uint32.createPointer(), "_trg_spk_event", "trgSpkEvent",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedTrgSpikeEventArray(g, "SpkEvent"); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_trg_st", "trgST",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedTrgSpikeArray(g, "ST"); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_trg_prev_st", "trgPrevST",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedTrgSpikeArray(g, "PrevST"); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_trg_set", "trgSET",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedTrgSpikeEventArray(g, "SET"); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_trg_prev_set", "trgPrevSET",
                 [](const auto &runtime, const auto &g, size_t) { return runtime.getFusedTrgSpikeEventArray(g, "PrevSET"); });

    // Connectivity fields
    if(env.getGroup().getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        env.addField(Uint32.createPointer(), "_gp", "gp",
                     [](const auto &runtime, const auto &sg, size_t) { return runtime.getArray(sg, "gp"); });
    }
    else if(env.getGroup().getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        env.addField(Uint32.createPointer(), "_row_length", "rowLength",
                     [](const auto &runtime, const auto &sg, size_t) { return runtime.getArray(sg, "rowLength"); });
        env.addField(env.getGroup().getArchetype().getSparseIndType().createPointer(), "_ind", "ind",
                     [](const auto &runtime, const auto &sg, size_t) { return runtime.getArray(sg, "ind"); });
        env.addField(Uint32.createPointer(), "_col_length", "colLength", 
                     [](const auto &runtime, const auto &sg, size_t) { return runtime.getArray(sg, "colLength"); });
        env.addField(Uint32.createPointer(), "_remap", "remap", 
                     [](const auto &runtime, const auto &sg, size_t) { return runtime.getArray(sg, "remap"); });
    }
    else if(env.getGroup().getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL) {
        // **TODO** automatic heterogeneity detection on all fields would make this much nicer
        std::ostringstream kernSizeInit;
        kernSizeInit << "const unsigned int kernelSize = ";
        const auto &kernelSize = env.getGroup().getArchetype().getKernelSize();
        for (size_t d = 0; d < kernelSize.size(); d++) {
            // If this dimension has a heterogeneous size, add it to struct
            if (isKernelSizeHeterogeneous(env.getGroup(), d)) {
                env.addField(Type::Uint32.addConst(), "_kernel_size_" + std::to_string(d), 
                             Type::Uint32, "kernelSize" + std::to_string(d),
                             [d](const auto&, const auto &g, size_t) { return g.getKernelSize().at(d); });
            }

            // Multiply size by dimension
            kernSizeInit << getKernelSize(env.getGroup(), d);
            if (d != (kernelSize.size() - 1)) {
                kernSizeInit << " * ";
            }
        }

        // Add size field
        kernSizeInit << ";";
        env.add(Type::Uint32.addConst(), "_kernel_size", "kernelSize",
                {env.addInitialiser(kernSizeInit.str())});
    }

    // If batching is enabled
    if(batchSize > 1) {
        // Calculate batch offsets into pre and postsynaptic populations
        env.add(Uint32.addConst(), "_pre_batch_offset", "preBatchOffset",
                {env.addInitialiser("const unsigned int preBatchOffset = $(num_pre) * $(batch);")});
        env.add(Uint32.addConst(), "_post_batch_offset", "postBatchOffset",
                {env.addInitialiser("const unsigned int postBatchOffset = $(num_post) * $(batch);")});
        
        // Calculate batch offsets into synapse arrays
        const auto indexType = backend.getSynapseIndexType(env.getGroup());
        const auto indexTypeName = indexType.getName();
        env.add(indexType.addConst(), "_syn_batch_offset", "synBatchOffset",
                {env.addInitialiser("const " + indexTypeName + " synBatchOffset = (" + indexTypeName + ")$(_pre_batch_offset) * $(_row_stride);")});
        
        // If group has kernel weights, calculate batch stride over them
        if(env.getGroup().getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL) {
            env.add(Uint32.addConst(), "_kern_batch_offset", "kernBatchOffset",
                    {env.addInitialiser("const unsigned int kernBatchOffset = $(_kernel_size) * $(batch);")});
        }
    }

    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
    if(env.getGroup().getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        const unsigned int numDelaySteps = env.getGroup().getArchetype().getDelaySteps();
        const unsigned int numSrcDelaySlots = env.getGroup().getArchetype().getSrcNeuronGroup()->getNumDelaySlots();

        std::ostringstream preDelaySlotInit;
        preDelaySlotInit << "const unsigned int preDelaySlot = ";
        if(numDelaySteps == 0) {
            preDelaySlotInit << "*$(_src_spk_que_ptr);" << std::endl;
        }
        else {
            preDelaySlotInit << "(*$(_src_spk_que_ptr) + " << (numSrcDelaySlots - numDelaySteps) << ") % " << numSrcDelaySlots <<  ";" << std::endl;
        }
        env.add(Uint32, "_pre_delay_slot", "preDelaySlot", 
                {env.addInitialiser(preDelaySlotInit.str())});

        env.add(Uint32, "_pre_delay_offset", "preDelayOffset",
                {env.addInitialiser("const unsigned int preDelayOffset = $(_pre_delay_slot) * $(num_pre);")});

        if(batchSize > 1) {
            env.add(Uint32, "_pre_batch_delay_slot", "preBatchDelaySlot",
                    {env.addInitialiser("const unsigned int preBatchDelaySlot = $(_pre_delay_slot) + ($(batch) * " + std::to_string(numSrcDelaySlots) + ");")});
            env.add(Uint32, "_pre_batch_delay_offset", "preBatchDelayOffset",
                    {env.addInitialiser("const unsigned int preBatchDelayOffset = $(_pre_delay_offset) + ($(_pre_batch_offset) * " + std::to_string(numSrcDelaySlots) + ");")});
        }

        env.add(Uint32, "_pre_prev_spike_time_delay_offset", "prePrevSpikeTimeDelayOffset",
                {env.addInitialiser("const unsigned int prePrevSpikeTimeDelayOffset = ((*$(_src_spk_que_ptr) + " 
                                    + std::to_string(numSrcDelaySlots - numDelaySteps - 1) + ") % " + std::to_string(numSrcDelaySlots) + ") * $(num_pre);")});

        if(batchSize > 1) {
            env.add(Uint32, "_pre_prev_spike_time_batch_delay_offset", "prePrevSpikeTimeBatchDelayOffset",
                    {env.addInitialiser("const unsigned int prePrevSpikeTimeBatchDelayOffset = $(_pre_prev_spike_time_delay_offset) + ($(_pre_batch_offset) * " + std::to_string(numSrcDelaySlots) + ");")});
        }
    }

    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
    if(env.getGroup().getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
        const unsigned int numBackPropDelaySteps = env.getGroup().getArchetype().getBackPropDelaySteps();
        const unsigned int numTrgDelaySlots = env.getGroup().getArchetype().getTrgNeuronGroup()->getNumDelaySlots();

        std::ostringstream postDelaySlotInit;
        postDelaySlotInit << "const unsigned int postDelaySlot = ";
        if(numBackPropDelaySteps == 0) {
            postDelaySlotInit << "*$(_trg_spk_que_ptr);" << std::endl;
        }
        else {
            postDelaySlotInit << "(*$(_trg_spk_que_ptr) + " << (numTrgDelaySlots - numBackPropDelaySteps) << ") % " << numTrgDelaySlots << ";" << std::endl;
        }
        env.add(Uint32, "_post_delay_slot", "postDelaySlot", 
                {env.addInitialiser(postDelaySlotInit.str())});

        env.add(Uint32, "_post_delay_offset", "postDelayOffset",
                {env.addInitialiser("const unsigned int postDelayOffset = $(_post_delay_slot) * $(num_post);")});

        if(batchSize > 1) {
            env.add(Uint32, "_post_batch_delay_slot", "postBatchDelaySlot",
                    {env.addInitialiser("const unsigned int postBatchDelaySlot =$(_post_delay_slot) + ($(batch) * " + std::to_string(numTrgDelaySlots) + ");")});
            env.add(Uint32, "_post_batch_delay_offset", "postBatchDelayOffset",
                    {env.addInitialiser("const unsigned int postBatchDelayOffset = $(_post_delay_offset) + ($(_post_batch_offset) * " + std::to_string(numTrgDelaySlots) + ");")});
        }

        env.add(Uint32, "_post_prev_spike_time_delay_offset", "postPrevSpikeTimeDelayOffset",
                {env.addInitialiser("const unsigned int postPrevSpikeTimeDelayOffset = ((*$(_trg_spk_que_ptr) + " 
                                    + std::to_string(numTrgDelaySlots - numBackPropDelaySteps - 1) + ") % " + std::to_string(numTrgDelaySlots) + ") * $(num_post);")});

        if(batchSize > 1) {
            env.add(Uint32, "_post_prev_spike_time_batch_delay_offset", "postPrevSpikeTimeBatchDelayOffset",
                    {env.addInitialiser("const unsigned int postPrevSpikeTimeBatchDelayOffset = $(_post_prev_spike_time_delay_offset) + ($(_post_batch_offset) * " + std::to_string(numTrgDelaySlots) + ");")});
        }
    }
}
//--------------------------------------------------------------------------
template<typename G>
void buildStandardCustomUpdateEnvironment(EnvironmentGroupMergedField<G> &env, unsigned int batchSize)
{
    // If batching is enabled, calculate batch offset
    const bool batched = (env.getGroup().getArchetype().getDims() & VarAccessDim::BATCH) && (batchSize > 1);
    if(batched) {
        env.add(Type::Uint32.addConst(), "_batch_offset", "batchOffset",
                {env.addInitialiser("const unsigned int batchOffset = $(size) * $(batch);")});
    }
            
    // If axonal delays are required
    if(env.getGroup().getArchetype().getDelayNeuronGroup() != nullptr) {
        // Add spike queue pointer field
        env.addField(Type::Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr", 
                     [](const auto &runtime, const auto &cg, size_t) 
                     { 
                         return runtime.getArray(*cg.getDelayNeuronGroup(), "spkQuePtr"); 
                     });

        // We should read from delay slot pointed to be spkQuePtr 
        env.add(Type::Uint32.addConst(), "_delay_slot", "delaySlot",
                {env.addInitialiser("const unsigned int delaySlot = * $(_spk_que_ptr);")});
        env.add(Type::Uint32.addConst(), "_delay_offset", "delayOffset",
                {env.addInitialiser("const unsigned int delayOffset = $(_delay_slot) * $(size);")});

        // If batching is also enabled, calculate offset including delay and batch
        if(batched) {
            const std::string numDelaySlotsStr = std::to_string(env.getGroup().getArchetype().getDelayNeuronGroup()->getNumDelaySlots());
            env.add(Type::Uint32.addConst(), "_batch_delay_slot", "batchDelaySlot",
                    {env.addInitialiser("const unsigned int batchDelaySlot = ($(batch) * " + numDelaySlotsStr + ") + $(_delay_slot);")});

            // Calculate current batch offset
            env.add(Type::Uint32.addConst(), "_batch_delay_offset", "batchDelayOffset",
                    {env.addInitialiser("const unsigned int batchDelayOffset = $(_delay_offset) + ($(_batch_offset) * " + numDelaySlotsStr + ");")});
        }
    }
}
//--------------------------------------------------------------------------
template<typename G>
void buildStandardCustomUpdateWUEnvironment(EnvironmentGroupMergedField<G> &env, unsigned int batchSize)
{
    // Add batch offset if group is batched
    if((env.getGroup().getArchetype().getDims() & VarAccessDim::BATCH) && (batchSize > 1)) {
        env.add(Type::Uint32.addConst(), "_batch_offset", "batchOffset",
                {env.addInitialiser("const unsigned int batchOffset = $(_size) * $(batch);")});
    }
}
//--------------------------------------------------------------------------
template<typename G>
void buildStandardCustomConnectivityUpdateEnvironment(const BackendBase &backend, EnvironmentGroupMergedField<G> &env)
{
    // Add fields for number of pre and postsynaptic neurons
    env.addField(Type::Uint32.addConst(), "num_pre",
                 Type::Uint32, "numSrcNeurons", 
                 [](const auto&, const auto &cg, size_t) 
                 { 
                     const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                     return sgInternal->getSrcNeuronGroup()->getNumNeurons();
                 });
    env.addField(Type::Uint32.addConst(), "num_post",
                 Type::Uint32, "numTrgNeurons", 
                 [](const auto&, const auto &cg, size_t) 
                 { 
                     const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                     return sgInternal->getTrgNeuronGroup()->getNumNeurons();
                 });
    env.addField(Type::Uint32, "_row_stride", "rowStride", 
                 [&backend](const auto&, const auto &cg, size_t) { return backend.getSynapticMatrixRowStride(*cg.getSynapseGroup()); });
    
    // Connectivity fields
    auto *sg = env.getGroup().getArchetype().getSynapseGroup();
    if(sg->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        env.addField(Type::Uint32.createPointer(), "_row_length", "rowLength",
                     [](const auto &runtime, const auto &cg, size_t) { return runtime.getArray(*cg.getSynapseGroup(), "rowLength"); });
        env.addField(sg->getSparseIndType().createPointer(), "_ind", "ind",
                     [](const auto &runtime, const auto &cg, size_t) { return runtime.getArray(*cg.getSynapseGroup(), "ind"); });
    }

    // If there are delays on presynaptic variable references
    if(env.getGroup().getArchetype().getPreDelayNeuronGroup() != nullptr) {
        env.addField(Type::Uint32.createPointer(), "_pre_spk_que_ptr", "preSpkQuePtr",
                     [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(*g.getPreDelayNeuronGroup(), "spkQuePtr"); });
    
        env.add(Type::Uint32.addConst(), "_pre_delay_offset", "preDelayOffset",
                {env.addInitialiser("const unsigned int preDelayOffset = (*$(_pre_spk_que_ptr) * $(num_pre));")});
    }
    
    // If there are delays on postsynaptic variable references
    if(env.getGroup().getArchetype().getPostDelayNeuronGroup() != nullptr) {
        env.addField(Type::Uint32.createPointer(), "_post_spk_que_ptr", "postSpkQuePtr",
                     [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(*g.getPostDelayNeuronGroup(), "spkQuePtr"); });

        env.add(Type::Uint32.addConst(), "_post_delay_offset", "postDelayOffset",
                {env.addInitialiser("const unsigned int postDelayOffset = (*$(_post_spk_que_ptr) * $(num_post));")});
    }
}
//-----------------------------------------------------------------------
template<typename G, typename S>
Type::ResolvedType getSynapseIndexType(const BackendBase &backend, const GroupMerged<G> &m,
                                       S getSynapseGroupFn)
{
    // If any merged groups have more synapses than can be represented using a uint32, use Uint64
    if(std::any_of(m.getGroups().cbegin(), m.getGroups().cend(),
                   [getSynapseGroupFn, &backend](const auto &g)
                   {
                       const auto &sg = getSynapseGroupFn(g.get());
                       const size_t numSynapses = (size_t)sg.getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(sg);
                       return (numSynapses > std::numeric_limits<uint32_t>::max());
                   }))
    {
        return Type::Uint64;
    }
    // Otherwise, use Uint64
    else {
        return Type::Uint32;
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::BackendBase
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
BackendBase::BackendBase(const PreferencesBase &preferences)
:   m_PointerBytes(sizeof(char *)), m_Preferences(preferences)
{
}
//-----------------------------------------------------------------------
Type::ResolvedType BackendBase::getSynapseIndexType(const GroupMerged<SynapseGroupInternal> &sg) const
{
    return ::getSynapseIndexType(*this, sg, 
                                 [](const auto &g)->const SynapseGroupInternal&
                                 { 
                                    return g; 
                                 });
}
//-----------------------------------------------------------------------
Type::ResolvedType BackendBase::getSynapseIndexType(const GroupMerged<CustomUpdateWUInternal> &cg) const
{
    return ::getSynapseIndexType(*this, cg, 
                                 [](const auto &g)->const SynapseGroupInternal&
                                 { 
                                    return *(g.getSynapseGroup()); 
                                 });
}
//-----------------------------------------------------------------------
Type::ResolvedType BackendBase::getSynapseIndexType(const GroupMerged<CustomConnectivityUpdateInternal> &cg) const
{
    return ::getSynapseIndexType(*this, cg, 
                                 [](const auto &g)->const SynapseGroupInternal&
                                 { 
                                    return *(g.getSynapseGroup()); 
                                 });
}
//-----------------------------------------------------------------------
void BackendBase::buildSizeEnvironment(EnvironmentGroupMergedField<CustomUpdateGroupMerged> &env) const
{
    buildCustomUpdateSizeEnvironment(env);
}
//-----------------------------------------------------------------------
void BackendBase::buildSizeEnvironment(EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> &env) const
{
    buildCustomUpdateWUSizeEnvironment(*this, env);
}
//-----------------------------------------------------------------------
void BackendBase::buildSizeEnvironment(EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> &env) const
{
    buildCustomUpdateWUSizeEnvironment(*this, env);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<NeuronUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardNeuronEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<NeuronPrevSpikeTimeUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardNeuronEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<NeuronSpikeQueueUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardNeuronEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<PresynapticUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<PostsynapticUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<SynapseDynamicsGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<SynapseDendriticDelayUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardCustomUpdateEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateWUGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardCustomUpdateWUEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardCustomUpdateWUEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env) const
{
    buildStandardCustomConnectivityUpdateEnvironment(*this, env);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<NeuronInitGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardNeuronEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<SynapseInitGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateInitGroupMerged> &env, unsigned int batchSize) const
{
    buildCustomUpdateSizeEnvironment(env);
    buildStandardCustomUpdateEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomWUUpdateInitGroupMerged> &env, unsigned int batchSize) const
{
    buildCustomUpdateWUSizeEnvironment(*this, env);
    buildStandardCustomUpdateWUEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomWUUpdateSparseInitGroupMerged> &env, unsigned int batchSize) const
{
    buildCustomUpdateWUSizeEnvironment(*this, env);
    buildStandardCustomUpdateWUEnvironment(env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdatePreInitGroupMerged> &env) const
{
    env.addField(Type::Uint32.addConst(), "size", 
                 Type::Uint32, "size",
                 [](const auto &, const auto &c, size_t) 
                 { 
                     return c.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); 
                 });

}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdatePostInitGroupMerged> &env) const
{
    env.addField(Type::Uint32.addConst(), "size", 
                 Type::Uint32, "size",
                 [](const auto &, const auto &c, size_t) 
                 { 
                     return c.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(); 
                 });
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<SynapseSparseInitGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdateSparseInitGroupMerged> &env) const
{
    buildStandardCustomConnectivityUpdateEnvironment(*this, env);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<SynapseConnectivityInitGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
}
//----------------------------------------------------------------------------
std::string BackendBase::getReductionInitialValue(VarAccessMode access, const Type::ResolvedType &type) const
{
    // If reduction is a sum, initialise to zero
    if(access & VarAccessModeAttribute::SUM) {
        return Type::writeNumeric(0, type);
    }
    // Otherwise, reduction is a maximum operation, return lowest value for type
    else if(access & VarAccessModeAttribute::MAX) {
        return Type::writeNumeric(type.getNumeric().lowest, type);
    }
    else {
        assert(false);
        return "";
    }
}
//----------------------------------------------------------------------------
std::string BackendBase::getReductionOperation(const std::string &reduction, const std::string &value,
                                               VarAccessMode access, const Type::ResolvedType &type) const
{
    // If operation is sum, add output of custom update to sum
    assert(type.isNumeric());
    if(access & VarAccessModeAttribute::SUM) {
        return reduction + " += " + value;
    }
    // Otherwise, if it's max
    else if(access & VarAccessModeAttribute::MAX) {
        // If type is integral, generate max call
        if(type.getNumeric().isIntegral) {
            return reduction + " = " + "max(" + reduction + ", " + value + ")";
            
        }
        // Otherwise, generate gmax call
        else {
            return reduction + " = " + "fmax(" + reduction + ", " + value + ")";
        }
    }
    else {
        assert(false);
        return "";
    }
}
//-----------------------------------------------------------------------
std::vector<BackendBase::ReductionTarget> BackendBase::genInitReductionTargets(CodeStream &os, const CustomUpdateGroupMerged &cg, 
                                                                               unsigned int batchSize, const std::string &idx) const
{
    return genInitReductionTargets<VarAccess>(
        os, cg, batchSize, idx,
        [batchSize, &cg](const Models::VarReference &varRef, const std::string &index)
        {
            return cg.getVarRefIndex(varRef.getDelayNeuronGroup() != nullptr, batchSize,
                                     varRef.getVarDims(), index);
        });
}
//-----------------------------------------------------------------------
std::vector<BackendBase::ReductionTarget> BackendBase::genInitReductionTargets(CodeStream &os, const CustomUpdateWUGroupMerged &cg, 
                                                                               unsigned int batchSize, const std::string &idx) const
{
    return genInitReductionTargets<VarAccess>(
        os, cg, batchSize, idx,
        [batchSize, &cg](const Models::WUVarReference &varRef, const std::string &index)
        {
            return cg.getVarRefIndex(batchSize, varRef.getVarDims(), index);
        });
}
}   // namespace GeNN::CodeGenerator