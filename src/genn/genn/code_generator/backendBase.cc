#include "code_generator/backendBase.h"

// GeNN includes
#include "gennUtils.h"
#include "logging.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"
#include "code_generator/customConnectivityUpdateGroupMerged.h"
#include "code_generator/customUpdateGroupMerged.h"
#include "code_generator/initGroupMerged.h"
#include "code_generator/neuronUpdateGroupMerged.h"
#include "code_generator/synapseUpdateGroupMerged.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
template<typename G>
void buildStandardNeuronEnvironment(const BackendBase &backend, EnvironmentGroupMergedField<G> &env, unsigned int batchSize)
{
    using namespace Type;

    env.addField(Uint32.addConst(), "num_neurons",
                 Uint32, "numNeurons",
                 [](const auto &ng, size_t) { return std::to_string(ng.getNumNeurons()); });
    env.addField(Uint32.createPointer(), "_spk_cnt", "spkCnt",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpkCnt" + g.getName(); });
    env.addField(Uint32.createPointer(), "_spk", "spk",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpk" + g.getName(); });
    env.addField(Uint32.createPointer(), "_spk_cnt_evnt", "spkCntEvnt",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpkCntEvnt" + g.getName(); });
    env.addField(Uint32.createPointer(), "_spk_evnt", "spkEvnt",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpkEvnt" + g.getName(); });
    env.addField(Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr",
                  [&backend](const auto &g, size_t) { return backend.getScalarAddressPrefix() + "spkQuePtr" + g.getName(); });

    env.addField(env.getGroup().getTimeType().createPointer(), "_spk_time", "sT",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "sT" + g.getName(); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_spk_evnt_time", "seT",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "seT" + g.getName(); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_prev_spk_time", "prevST", 
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "prevST" + g.getName(); });
    env.addField(env.getGroup().getTimeType().createPointer(), "_prev_spk_evnt_time", "prevSET",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "prevSET" + g.getName(); });

    env.addField(Uint32.createPointer(), "_record_spk", "recordSpk",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "recordSpk" + g.getName(); }, 
                 "", GroupMergedFieldType::DYNAMIC);
    env.addField(Uint32.createPointer(), "_record_spk_event", "recordSpkEvent",
                 [&backend](const auto &g, size_t){ return backend.getDeviceVarPrefix() + "recordSpkEvent" + g.getName(); },
                 "", GroupMergedFieldType::DYNAMIC);

    // If batching is enabled, calculate batch offset
    if(batchSize > 1) {
        env.add(Uint32.addConst(), "_batchOffset", "batchOffset",
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
                {env.addInitialiser("const unsigned int writeDelaySlot = * $(_spk_que_ptr);")});
        env.add(Uint32.addConst(), "_write_delay_offset", "writeDelayOffset",
                {env.addInitialiser("const unsigned int writeDelayOffset = $(_write_delay_slot) * $(num_neurons);")});

        // If batching is also enabled
        if(batchSize > 1) {
            // Calculate batched delay slots
            env.add(Uint32.addConst(), "_read_batch_delay_slot", "readBatchDelaySlot",
                    {env.addInitialiser("const unsigned int readBatchDelaySlot = (batch * " + numDelaySlotsStr + ") + $(_read_delay_slot);")});
            env.add(Uint32.addConst(), "_write_batch_delay_slot", "writeBatchDelaySlot",
                    {env.addInitialiser("const unsigned int writeBatchDelaySlot = (batch * " + numDelaySlotsStr + ") + $(_write_delay_slot);")});

            // Calculate current batch offset
            env.add(Uint32.addConst(), "_batch_delay_offset", "batchDelayOffset",
                    {env.addInitialiser("const unsigned int batchDelayOffset = $(_batch_offset) * " + numDelaySlotsStr + ";")});

            // Calculate further offsets to include delay and batch
            env.add(Uint32.addConst(), "_read_batch_delay_offset", "readBatchDelayOffset",
                    {env.addInitialiser("const unsigned int readBatchDelayOffset = $(_read_delay_offset) + $(_batch_delay_offset);")});
            env.add(Uint32.addConst(), "_write_batch_delay_offset", "writeBatchDelayOffset",
                    {env.addInitialiser("const unsigned int writeBatchDelayOffset = $(_write_delay_offset)+ $(_batch_delay_offset);")});
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
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
    env.addField(Uint32.addConst(), "num_post",
                 Uint32, "numTrgNeurons", 
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
    env.addField(Uint32, "_row_stride", "rowStride", 
                 [&backend](const SynapseGroupInternal &sg, size_t) { return std::to_string(backend.getSynapticMatrixRowStride(sg)); });
    env.addField(Uint32, "_col_stride", "colStride", 
                 [](const SynapseGroupInternal &sg, size_t) { return std::to_string(sg.getMaxSourceConnections()); });
                        
    // Postsynaptic model fields         
    env.addField(env.getGroup().getScalarType().createPointer(), "_out_post", "outPost",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "outPost" + g.getFusedPSVarSuffix(); });
    env.addField(env.getGroup().getScalarType().createPointer(), "_den_delay", "denDelay",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "denDelay" + g.getFusedPSVarSuffix(); });
    env.addField(Uint32.createPointer(), "_den_delay_ptr", "denDelayPtr",
                 [&backend](const auto &g, size_t) { return backend.getScalarAddressPrefix() + "denDelayPtr" + g.getFusedPSVarSuffix(); });
                       
    // Presynaptic output fields
    env.addField(env.getGroup().getScalarType().createPointer(), "_out_pre", "outPre",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "outPre" + g.getFusedPreOutputSuffix(); });
                        
    // Source neuron fields
    env.addField(Uint32.createPointer(), "_src_spk_que_ptr", "srcSpkQuePtr",
                 [&backend](const auto &g, size_t) { return backend.getScalarAddressPrefix() + "spkQuePtr" + g.getSrcNeuronGroup()->getName(); });
    env.addField(Uint32.createPointer(), "_src_spk_cnt", "srcSpkCnt",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpkCnt" + g.getSrcNeuronGroup()->getName(); });
    env.addField(Uint32.createPointer(), "_src_spk", "srcSpk",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpk" + g.getSrcNeuronGroup()->getName(); });
    env.addField(Uint32.createPointer(), "_src_spk_evnt_cnt", "srcSpkCntEvnt",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpkCntEvnt" + g.getSrcNeuronGroup()->getName(); });
    env.addField(Uint32.createPointer(), "_src_spk_evnt", "srcSpkEvnt",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpkEvnt" + g.getSrcNeuronGroup()->getName(); });

    // Target neuron fields
    env.addField(Uint32.createPointer(), "_trg_spk_que_ptr", "trgSpkQuePtr",
                 [&backend](const auto &g, size_t) { return backend.getScalarAddressPrefix() + "spkQuePtr" + g.getTrgNeuronGroup()->getName(); });
    env.addField(Uint32.createPointer(), "_trg_spk_cnt", "trgSpkCnt",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpkCnt" + g.getTrgNeuronGroup()->getName(); });
    env.addField(Uint32.createPointer(), "_trg_spk", "trgSpk",
                 [&backend](const auto &g, size_t) { return backend.getDeviceVarPrefix() + "glbSpk" + g.getTrgNeuronGroup()->getName(); });
        
    // Connectivity fields
    if(env.getGroup().getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
        env.addField(Uint32.createPointer(), "_gp", "gp",
                     [&backend](const auto &sg, size_t) { return backend.getDeviceVarPrefix() + "gp" + sg.getName(); });
    }
    else if(env.getGroup().getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
        env.addField(Uint32.createPointer(), "_row_length", "rowLength",
                     [&backend](const auto &sg, size_t) { return backend.getDeviceVarPrefix() + "rowLength" + sg.getName(); });
        env.addField(env.getGroup().getArchetype().getSparseIndType().createPointer(), "_ind", "ind",
                     [&backend](const auto &sg, size_t) { return backend.getDeviceVarPrefix() + "ind" + sg.getName(); });
        env.addField(Uint32.createPointer(), "_col_length", "colLength", 
                     [&backend](const auto &sg, size_t) { return backend.getDeviceVarPrefix() + "colLength" + sg.getName(); });
        env.addField(Uint32.createPointer(), "_remap", "remap", 
                     [&backend](const auto &sg, size_t) { return backend.getDeviceVarPrefix() + "remap" + sg.getName(); });
    }

    // If batching is enabled
    if(batchSize > 1) {
        // Calculate batch offsets into pre and postsynaptic populations
        env.add(Uint32.addConst(), "_pre_batch_offset", "preBatchOffset",
                {env.addInitialiser("const unsigned int preBatchOffset = $(num_pre) * $(batch);")});
        env.add(Uint32.addConst(), "_post_batch_offset", "postBatchOffset",
                {env.addInitialiser("const unsigned int preBatchOffset = $(num_post) * $(batch);")});
        
        // Calculate batch offsets into synapse arrays, using 64-bit arithmetic if necessary
        if(backend.areSixtyFourBitSynapseIndicesRequired(env.getGroup())) {
            assert(false);
            //os << "const uint64_t synBatchOffset = (uint64_t)preBatchOffset * (uint64_t)group->rowStride;" << std::endl;
        }
        else {
            env.add(Uint32.addConst(), "_syn_batch_offset", "synBatchOffset",
                    {env.addInitialiser("const unsigned int synBatchOffset = $(_pre_batch_offset) * $(_row_stride);")});
        }
        
        // If synapse group has kernel
        const auto &kernelSize = env.getGroup().getArchetype().getKernelSize();
        if(!kernelSize.empty()) {
            // Loop through kernel dimensions and multiply together
            // **TODO** extract list of kernel size variables referenced
            std::ostringstream kernBatchOffsetInit;
            kernBatchOffsetInit << "const unsigned int kernBatchOffset = ";
            for(size_t i = 0; i < kernelSize.size(); i++) {
                kernBatchOffsetInit << getKernelSize(env.getGroup(), i) << " * ";
            }
            
            // And finally by batch
            kernBatchOffsetInit << "$(batch);" << std::endl;

            env.add(Uint32.addConst(), "_kern_batch_offset", "kernBatchOffset",
                    {env.addInitialiser(kernBatchOffsetInit.str())});
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

        if(env.getGroup().getArchetype().getWUModel()->isPrevPreSpikeTimeRequired() 
            || env.getGroup().getArchetype().getWUModel()->isPrevPreSpikeEventTimeRequired()) 
        {
            env.add(Uint32, "_pre_prev_spike_time_delay_offset", "prePrevSpikeTimeDelayOffset",
                    {env.addInitialiser("const unsigned int prePrevSpikeTimeDelayOffset = ((*$(_src_spk_que_ptr) + " 
                                        + std::to_string(numSrcDelaySlots - numDelaySteps - 1) + ") % " + std::to_string(numSrcDelaySlots) + ") * $(num_pre);")});

            if(batchSize > 1) {
                env.add(Uint32, "_pre_prev_spike_time_batch_delay_offset", "prePrevSpikeTimeBatchDelayOffset",
                        {env.addInitialiser("const unsigned int prePrevSpikeTimeBatchDelayOffset = $(_pre_prev_spike_time_delay_offset) + ($(_pre_batch_offset) * " + std::to_string(numSrcDelaySlots) + ");")});
            }
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
                    {env.addInitialiser("const unsigned int postBatchDelaySlot =$(_post_delay_slot) + (batch * " + std::to_string(numTrgDelaySlots) + ");")});
            env.add(Uint32, "_post_batch_delay_offset", "postBatchDelayOffset",
                    {env.addInitialiser("const unsigned int postBatchDelayOffset = $(_post_delay_offset) + ($(_post_batch_offset) * " + std::to_string(numTrgDelaySlots) + ");")});
        }

        if(env.getGroup().getArchetype().getWUModel()->isPrevPostSpikeTimeRequired()) {
            env.add(Uint32, "_post_prev_spike_time_delay_offset", "postPrevSpikeTimeDelayOffset",
                    {env.addInitialiser("const unsigned int postPrevSpikeTimeDelayOffset = ((*$(_trg_spk_que_ptr) + " 
                                        + std::to_string(numTrgDelaySlots - numBackPropDelaySteps - 1) + ") % " + std::to_string(numTrgDelaySlots) + ") * $(num_post);")});

            if(batchSize > 1) {
                env.add(Uint32, "_post_prev_spike_time_batch_delay_offset", "postPrevSpikeTimeBatchDelayOffset",
                        {env.addInitialiser("const unsigned int postPrevSpikeTimeBatchDelayOffset = $(_post_prev_spike_time_delay_offset) + ($(_post_batch_offset) * " + std::to_string(numTrgDelaySlots) + ");")});
            }

        }
    }
}
}
//--------------------------------------------------------------------------
// GeNN::CodeGenerator::BackendBase
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
BackendBase::BackendBase(const PreferencesBase &preferences)
:   m_PointerBytes(sizeof(char *)), m_Preferences(preferences)
{
}
//--------------------------------------------------------------------------
bool BackendBase::areSixtyFourBitSynapseIndicesRequired(const GroupMerged<SynapseGroupInternal> &sg) const
{
    // Loop through merged groups and calculate maximum number of synapses
    size_t maxSynapses = 0;
    for(const auto &g : sg.getGroups()) {
        const size_t numSynapses = (size_t)g.get().getSrcNeuronGroup()->getNumNeurons() * (size_t)getSynapticMatrixRowStride(g.get());
        maxSynapses = std::max(maxSynapses, numSynapses);
    }

    // Return true if any high bits are set
    return ((maxSynapses & 0xFFFFFFFF00000000ULL) != 0);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<NeuronUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardNeuronEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<NeuronPrevSpikeTimeUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardNeuronEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<NeuronSpikeQueueUpdateGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardNeuronEnvironment(*this, env, batchSize);
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
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomUpdateGroupMerged> &env) const
{
    // Add size field
    env.addField(Type::Uint32, "size", "size", 
                 [](const auto &c, size_t) { return std::to_string(c.getSize()); });
    
    // If batching is enabled, calculate batch offset
    if(env.getGroup().getArchetype().isBatched()) {
        env.add(Type::Uint32.addConst(), "_batch_offset", "batchOffset",
                {env.addInitialiser("const unsigned int batchOffset = $(size) * batch;")});
    }
            
    // If axonal delays are required
    if(env.getGroup().getArchetype().getDelayNeuronGroup() != nullptr) {
        // Add spike queue pointer field
        env.addField(Type::Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr", 
                     [this](const auto &cg, size_t) 
                     { 
                         return getScalarAddressPrefix() + "spkQuePtr" + cg.getDelayNeuronGroup()->getName(); 
                     });

        // We should read from delay slot pointed to be spkQuePtr 
        env.add(Type::Uint32.addConst(), "_delay_slot", "delaySlot",
                {env.addInitialiser("const unsigned int delaySlot = * $(_spk_que_ptr);")});
        env.add(Type::Uint32.addConst(), "_delay_offset", "delayOffset",
                {env.addInitialiser("const unsigned int delayOffset = $(_delay_slot) * $(size);")});

        // If batching is also enabled, calculate offset including delay and batch
        if(env.getGroup().getArchetype().isBatched()) {
            const std::string numDelaySlotsStr = std::to_string(env.getGroup().getArchetype().getDelayNeuronGroup()->getNumDelaySlots());
            env.add(Type::Uint32.addConst(), "_batch_delay_slot", "batchDelaySlot",
                    {env.addInitialiser("const unsigned int batchDelaySlot = (batch * " + numDelaySlotsStr + ") + $(_delay_slot);")});

            // Calculate current batch offset
            env.add(Type::Uint32.addConst(), "_batch_delay_offset", "batchDelayOffset",
                    {env.addInitialiser("const unsigned int batchDelayOffset = $(_batch_offset) * " + numDelaySlotsStr + ";")});
        }
    }
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env) const
{
    // If there are delays on presynaptic variable references
    if(env.getGroup().getArchetype().getPreDelayNeuronGroup() != nullptr) {
        env.add(Type::Uint32.addConst(), "_pre_delay_offset", "preDelayOffset",
                {env.addInitialiser("const unsigned int preDelayOffset = (*$(_pre_spk_que_ptr) * $(num_pre));")});
    }
    
    // If there are delays on postsynaptic variable references
    if(env.getGroup().getArchetype().getPostDelayNeuronGroup() != nullptr) {
        env.add(Type::Uint32.addConst(), "_post_delay_offset", "postDelayOffset",
                {env.addInitialiser("const unsigned int postDelayOffset = (*$(_post_spk_que_ptr) * $(num_post));")});
    }
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<NeuronInitGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardNeuronEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<SynapseInitGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
}
//-----------------------------------------------------------------------
void BackendBase::buildStandardEnvironment(EnvironmentGroupMergedField<SynapseSparseInitGroupMerged> &env, unsigned int batchSize) const
{
    buildStandardSynapseEnvironment(*this, env, batchSize);
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
    assert(type.isNumeric());
    if(access & VarAccessModeAttribute::SUM) {
        return "0";
    }
    // Otherwise, reduction is a maximum operation, return lowest value for type
    else if(access & VarAccessModeAttribute::MAX) {
        return Utils::writePreciseString(type.getNumeric().lowest);
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
std::vector<BackendBase::ReductionTarget> BackendBase::genInitReductionTargets(CodeStream &os, const CustomUpdateGroupMerged &cg, const std::string &idx) const
{
    return genInitReductionTargets(os, cg, idx,
                                   [&cg](const Models::VarReference &varRef, const std::string &index)
                                   {
                                       return cg.getVarRefIndex(varRef.getDelayNeuronGroup() != nullptr,
                                                                getVarAccessDuplication(varRef.getVar().access),
                                                                index);
                                   });
}
//-----------------------------------------------------------------------
std::vector<BackendBase::ReductionTarget> BackendBase::genInitReductionTargets(CodeStream &os, const CustomUpdateWUGroupMerged &cg, const std::string &idx) const
{
    return genInitReductionTargets(os, cg, idx,
                                   [&cg](const Models::WUVarReference &varRef, const std::string &index)
                                   {
                                       return cg.getVarRefIndex(getVarAccessDuplication(varRef.getVar().access),
                                                                index);
                                   });
}
}   // namespace GeNN::CodeGenerator