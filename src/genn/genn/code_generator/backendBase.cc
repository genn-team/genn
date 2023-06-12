#include "code_generator/backendBase.h"

// GeNN includes
#include "gennUtils.h"
#include "logging.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"
#include "code_generator/customConnectivityUpdateGroupMerged.h"
#include "code_generator/customUpdateGroupMerged.h"
#include "code_generator/neuronUpdateGroupMerged.h"

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
bool BackendBase::areSixtyFourBitSynapseIndicesRequired(const SynapseGroupMergedBase &sg) const
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
void BackendBase::genNeuronIndexCalculation(CodeStream &os, const NeuronUpdateGroupMerged &ng, unsigned int batchSize) const
{
    // If batching is enabled, calculate batch offset
    if(batchSize > 1) {
        os << "const unsigned int batchOffset = group->numNeurons * batch;" << std::endl;
    }
            
    // If axonal delays are required
    if(ng.getArchetype().isDelayRequired()) {
        // We should READ from delay slot before spkQuePtr
        os << "const unsigned int readDelaySlot = (*group->spkQuePtr + " << (ng.getArchetype().getNumDelaySlots() - 1) << ") % " << ng.getArchetype().getNumDelaySlots() << ";" << std::endl;
        os << "const unsigned int readDelayOffset = readDelaySlot * group->numNeurons;" << std::endl;

        // And we should WRITE to delay slot pointed to be spkQuePtr
        os << "const unsigned int writeDelaySlot = *group->spkQuePtr;" << std::endl;
        os << "const unsigned int writeDelayOffset = writeDelaySlot * group->numNeurons;" << std::endl;

        // If batching is also enabled
        if(batchSize > 1) {
            // Calculate batched delay slots
            os << "const unsigned int readBatchDelaySlot = (batch * " << ng.getArchetype().getNumDelaySlots() << ") + readDelaySlot;" << std::endl;
            os << "const unsigned int writeBatchDelaySlot = (batch * " << ng.getArchetype().getNumDelaySlots() << ") + writeDelaySlot;" << std::endl;

            // Calculate current batch offset
            os << "const unsigned int batchDelayOffset = batchOffset * " << ng.getArchetype().getNumDelaySlots() << ";" << std::endl;

            // Calculate further offsets to include delay and batch
            os << "const unsigned int readBatchDelayOffset = readDelayOffset + batchDelayOffset;" << std::endl;
            os << "const unsigned int writeBatchDelayOffset = writeDelayOffset + batchDelayOffset;" << std::endl;
        }
    }
}
//-----------------------------------------------------------------------
void BackendBase::genSynapseIndexCalculation(EnvironmentExternal &env, const SynapseGroupMergedBase &sg, unsigned int batchSize) const
{
    // If batching is enabled
    if(batchSize > 1) {
        // Calculate batch offsets into pre and postsynaptic populations
        env.add(Type::Uint32.addConst(), "_pre_batch_offset", "preBatchOffset",
                {env.addInitialiser("const unsigned int preBatchOffset = " + env["num_pre"] + " * " + env["batch"] + ";")});
        env.add(Type::Uint32.addConst(), "_post_batch_offset", "postBatchOffset",
                {env.addInitialiser("const unsigned int preBatchOffset = " + env["num_post"] + " * " + env["batch"] + ";")});
        
        // Calculate batch offsets into synapse arrays, using 64-bit arithmetic if necessary
        if(areSixtyFourBitSynapseIndicesRequired(sg)) {
            assert(false);
            //os << "const uint64_t synBatchOffset = (uint64_t)preBatchOffset * (uint64_t)group->rowStride;" << std::endl;
        }
        else {
            env.add(Type::Uint32.addConst(), "_syn_batch_offset", "synBatchOffset",
                {env.addInitialiser("const unsigned int synBatchOffset = " + env["_pre_batch_offset"] + " * " + env["_row_stride"] + ";")});
        }
        
        // If synapse group has kernel weights
        /*const auto &kernelSize = sg.getArchetype().getKernelSize();
        if((sg.getArchetype().getMatrixType() & SynapseMatrixWeight::KERNEL) && !kernelSize.empty()) {
            // Loop through kernel dimensions and multiply together
            os << "const unsigned int kernBatchOffset = ";
            for(size_t i = 0; i < kernelSize.size(); i++) {
                os << sg.getKernelSize(i) << " * ";
            }
            
            // And finally by batch
            os << "batch;" << std::endl;
        }
    }

    // If presynaptic neuron group has variable queues, calculate offset to read from its variables with axonal delay
    if(sg.getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        const unsigned int numDelaySteps = sg.getArchetype().getDelaySteps();
        const unsigned int numSrcDelaySlots = sg.getArchetype().getSrcNeuronGroup()->getNumDelaySlots();

        os << "const unsigned int preDelaySlot = ";
        if(numDelaySteps == 0) {
            os << "*group->srcSpkQuePtr;" << std::endl;
        }
        else {
            os << "(*group->srcSpkQuePtr + " << (numSrcDelaySlots - numDelaySteps) << ") % " << numSrcDelaySlots <<  ";" << std::endl;
        }
        os << "const unsigned int preDelayOffset = preDelaySlot * group->numSrcNeurons;" << std::endl;

        if(batchSize > 1) {
            os << "const unsigned int preBatchDelaySlot = preDelaySlot + (batch * " << numSrcDelaySlots << ");" << std::endl;
            os << "const unsigned int preBatchDelayOffset = preDelayOffset + (preBatchOffset * " << numSrcDelaySlots << ");" << std::endl;
        }

        if(sg.getArchetype().getWUModel()->isPrevPreSpikeTimeRequired() || sg.getArchetype().getWUModel()->isPrevPreSpikeEventTimeRequired()) {
            os << "const unsigned int prePrevSpikeTimeDelayOffset = " << "((*group->srcSpkQuePtr + " << (numSrcDelaySlots - numDelaySteps - 1) << ") % " << numSrcDelaySlots << ")" << " * group->numSrcNeurons;" << std::endl;

            if(batchSize > 1) {
                os << "const unsigned int prePrevSpikeTimeBatchDelayOffset = prePrevSpikeTimeDelayOffset + (preBatchOffset * " << numSrcDelaySlots << ");" << std::endl;
            }
        }
    }

    // If postsynaptic neuron group has variable queues, calculate offset to read from its variables at current time
    if(sg.getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
        const unsigned int numBackPropDelaySteps = sg.getArchetype().getBackPropDelaySteps();
        const unsigned int numTrgDelaySlots = sg.getArchetype().getTrgNeuronGroup()->getNumDelaySlots();

        os << "const unsigned int postDelaySlot = ";
        if(numBackPropDelaySteps == 0) {
            os << "*group->trgSpkQuePtr;" << std::endl;
        }
        else {
            os << "(*group->trgSpkQuePtr + " << (numTrgDelaySlots - numBackPropDelaySteps) << ") % " << numTrgDelaySlots << ";" << std::endl;
        }
        os << "const unsigned int postDelayOffset = postDelaySlot * group->numTrgNeurons;" << std::endl;

        if(batchSize > 1) {
            os << "const unsigned int postBatchDelaySlot = postDelaySlot + (batch * " << numTrgDelaySlots << ");" << std::endl;
            os << "const unsigned int postBatchDelayOffset = postDelayOffset + (postBatchOffset * " << numTrgDelaySlots << ");" << std::endl;
        }

        if(sg.getArchetype().getWUModel()->isPrevPostSpikeTimeRequired()) {
            os << "const unsigned int postPrevSpikeTimeDelayOffset = " << "((*group->trgSpkQuePtr + " << (numTrgDelaySlots - numBackPropDelaySteps - 1) << ") % " << numTrgDelaySlots << ")" << " * group->numTrgNeurons;" << std::endl;
            
            if(batchSize > 1) {
                os << "const unsigned int postPrevSpikeTimeBatchDelayOffset = postPrevSpikeTimeDelayOffset + (postBatchOffset * " << numTrgDelaySlots << ");" << std::endl;
            }

        }
    }*/
}
//-----------------------------------------------------------------------
void BackendBase::genCustomUpdateIndexCalculation(CodeStream &os, const CustomUpdateGroupMerged &cu) const
{
    // If batching is enabled, calculate batch offset
    if(cu.getArchetype().isBatched()) {
        os << "const unsigned int batchOffset = group->size * batch;" << std::endl;
    }
            
    // If axonal delays are required
    if(cu.getArchetype().getDelayNeuronGroup() != nullptr) {
        // We should read from delay slot pointed to be spkQuePtr
        os << "const unsigned int delaySlot = *group->spkQuePtr;" << std::endl;
        os << "const unsigned int delayOffset = (delaySlot * group->size);" << std::endl;

        // If batching is also enabled, calculate offset including delay and batch
        if(cu.getArchetype().isBatched()) {
            os << "const unsigned int batchDelaySlot = (batch * " << cu.getArchetype().getDelayNeuronGroup()->getNumDelaySlots() << ") + delaySlot;" << std::endl;

            // Calculate current batch offset
            os << "const unsigned int batchDelayOffset = delayOffset + (batchOffset * " << cu.getArchetype().getDelayNeuronGroup()->getNumDelaySlots() << ");" << std::endl;
        }
    }
}
//-----------------------------------------------------------------------
void BackendBase::genCustomConnectivityUpdateIndexCalculation(CodeStream &os, const CustomConnectivityUpdateGroupMerged &cu) const
{
    // If there are delays on presynaptic variable references
    if(cu.getArchetype().getPreDelayNeuronGroup() != nullptr) {
        os << "const unsigned int preDelayOffset = (*group->preSpkQuePtr * group->numSrcNeurons);" << std::endl;
    }
    
    // If there are delays on postsynaptic variable references
    if(cu.getArchetype().getPostDelayNeuronGroup() != nullptr) {
        os << "const unsigned int postDelayOffset = (*group->postSpkQuePtr * group->numTrgNeurons);" << std::endl;
    }
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