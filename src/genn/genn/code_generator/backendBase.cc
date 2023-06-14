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