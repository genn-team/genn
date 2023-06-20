#include "code_generator/groupMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/environment.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronSpikeQueueUpdateGroupMerged::name = "NeuronSpikeQueueUpdate";
//----------------------------------------------------------------------------
NeuronSpikeQueueUpdateGroupMerged::NeuronSpikeQueueUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                     const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   GroupMerged<NeuronGroupInternal>(index, typeContext, groups)
{
    using namespace Type;

    if(getArchetype().isDelayRequired()) {
        addPointerField(Uint32, "spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    } 

    addPointerField(Uint32, "spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField(Uint32, "spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
    }
}
//----------------------------------------------------------------------------
void NeuronSpikeQueueUpdateGroupMerged::genMergedGroupSpikeCountReset(EnvironmentExternalBase &env, unsigned int batchSize) const
{
    if(getArchetype().isSpikeEventRequired()) {
        if(getArchetype().isDelayRequired()) {
            env.getStream() << env["_spk_cnt_evnt"] << "[*" << env["_spk_que_ptr"];
            if(batchSize > 1) {
                env.getStream() << " + (batch * " << getArchetype().getNumDelaySlots() << ")";
            }
            env.getStream() << "] = 0; " << std::endl;
        }
        else {
            env.getStream() << env["_spk_cnt_evnt"] << "[" << ((batchSize > 1) ? "batch" : "0") << "] = 0;" << std::endl;
        }
    }

    if(getArchetype().isTrueSpikeRequired() && getArchetype().isDelayRequired()) {
        env.getStream() << env["_spk_cnt"] << "[*" << env["_spk_que_ptr"];
        if(batchSize > 1) {
            env.getStream() << " + (batch * " << getArchetype().getNumDelaySlots() << ")";
        }
        env.getStream() << "] = 0; " << std::endl;
    }
    else {
        env.getStream() << env["_spk_cnt"] << "[" << ((batchSize > 1) ? "batch" : "0") << "] = 0;" << std::endl;
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string NeuronPrevSpikeTimeUpdateGroupMerged::name = "NeuronPrevSpikeTimeUpdate";
//----------------------------------------------------------------------------
NeuronPrevSpikeTimeUpdateGroupMerged::NeuronPrevSpikeTimeUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                           const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
:   GroupMerged<NeuronGroupInternal>(index, typeContext, groups)
{
    using namespace Type;

    if(getArchetype().isDelayRequired()) {
        addPointerField(Uint32, "spkQuePtr", backend.getScalarAddressPrefix() + "spkQuePtr");
    } 

    addPointerField(Uint32, "spkCnt", backend.getDeviceVarPrefix() + "glbSpkCnt");

    if(getArchetype().isSpikeEventRequired()) {
        addPointerField(Uint32, "spkCntEvnt", backend.getDeviceVarPrefix() + "glbSpkCntEvnt");
    }

    if(getArchetype().isPrevSpikeTimeRequired()) {
        addPointerField(Uint32, "spk", backend.getDeviceVarPrefix() + "glbSpk");
        addPointerField(getTimeType(), "prevST", backend.getDeviceVarPrefix() + "prevST");
    }
    if(getArchetype().isPrevSpikeEventTimeRequired()) {
        addPointerField(Uint32, "spkEvnt", backend.getDeviceVarPrefix() + "glbSpkEvnt");
        addPointerField(getTimeType(), "prevSET", backend.getDeviceVarPrefix() + "prevSET");
    }

    if(getArchetype().isDelayRequired()) {
        addField(Uint32, "numNeurons",
                 [](const auto &ng, size_t) { return std::to_string(ng.getNumNeurons()); });
    }
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronGroupMergedBase
//----------------------------------------------------------------------------
bool NeuronGroupMergedBase::isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
{
    const auto *varInitSnippet = getArchetype().getVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUParamHeterogeneous(const std::string &paramName) const
{
    return (isWUParamReferenced(paramName) && 
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isWUParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getWUDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUGlobalVarHeterogeneous(const std::string &varName) const
{
    return (isWUGlobalVarReferenced(varName) &&
            isParamValueHeterogeneous(varName, [](const SynapseGroupInternal &sg) { return sg.getWUConstInitVals(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isWUVarInitParamReferenced(varName, paramName) &&
            isParamValueHeterogeneous(paramName, [varName](const SynapseGroupInternal &sg){ return sg.getWUVarInitialisers().at(varName).getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
{
    return (isWUVarInitParamReferenced(varName, paramName) && 
            isParamValueHeterogeneous(paramName, [varName](const SynapseGroupInternal &sg) { return sg.getWUVarInitialisers().at(varName).getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isSparseConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitParamHeterogeneous(const std::string &paramName) const
{
    return (isToeplitzConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isToeplitzConnectivityInitParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronParamHeterogeneous(const std::string &paramName) const
{
    return (isSrcNeuronParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isSrcNeuronParamReferenced(paramName) &&  
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getSrcNeuronGroup()->getDerivedParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronParamHeterogeneous(const std::string &paramName) const
{
    return (isTrgNeuronParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getParams(); }));
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronDerivedParamHeterogeneous(const std::string &paramName) const
{
    return (isTrgNeuronParamReferenced(paramName) &&
            isParamValueHeterogeneous(paramName, [](const SynapseGroupInternal &sg) { return sg.getTrgNeuronGroup()->getDerivedParams(); }));
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPreSlot(unsigned int batchSize) const
{
    if(getArchetype().getSrcNeuronGroup()->isDelayRequired()) {
        return  (batchSize == 1) ? "preDelaySlot" : "preBatchDelaySlot";
    }
    else {
        return (batchSize == 1) ? "0" : "batch";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostSlot(unsigned int batchSize) const
{
    if(getArchetype().getTrgNeuronGroup()->isDelayRequired()) {
        return  (batchSize == 1) ? "postDelaySlot" : "postBatchDelaySlot";
    }
    else {
        return (batchSize == 1) ? "0" : "batch";
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostDenDelayIndex(unsigned int batchSize, const std::string &index, const std::string &offset) const
{
    assert(getArchetype().isDendriticDelayRequired());

    const std::string batchID = ((batchSize == 1) ? "" : "postBatchOffset + ") + index;

    if(offset.empty()) {
        return "(*group->denDelayPtr * group->numTrgNeurons) + " + batchID;
    }
    else {
        return "(((*group->denDelayPtr + " + offset + ") % " + std::to_string(getArchetype().getMaxDendriticDelayTimesteps()) + ") * group->numTrgNeurons) + " + batchID;
    }
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPreVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    return getVarIndex(delay, batchSize, varDuplication, index, "pre");
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
   return getVarIndex(delay, batchSize, varDuplication, index, "post");
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
   
    if(delay) {
        return (singleBatch ? "prePrevSpikeTimeDelayOffset + " : "prePrevSpikeTimeBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "preBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
   
    if(delay) {
        return (singleBatch ? "postPrevSpikeTimeDelayOffset + " : "postPrevSpikeTimeBatchDelayOffset + ") + index;
    }
    else {
        return (singleBatch ? "" : "postBatchOffset + ") + index;
    }
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getSynVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "synBatchOffset + ") + index;
}
//--------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getKernelVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
{
    const bool singleBatch = (varDuplication == VarAccessDuplication::SHARED || batchSize == 1);
    return (singleBatch ? "" : "kernBatchOffset + ") + index;
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type SynapseGroupMergedBase::getHashDigest(Role role) const
{
    const bool updateRole = ((role == Role::PresynapticUpdate)
                             || (role == Role::PostsynapticUpdate)
                             || (role == Role::SynapseDynamics));

    // Update hash with archetype's hash
    boost::uuids::detail::sha1 hash;
    if(updateRole) {
        Utils::updateHash(getArchetype().getWUHashDigest(), hash);
    }
    else if (role == Role::ConnectivityInit) {
        Utils::updateHash(getArchetype().getConnectivityInitHashDigest(), hash);
    }
    else {
        Utils::updateHash(getArchetype().getWUInitHashDigest(), hash);
    }

    // Update hash with number of neurons in pre and postsynaptic population
    updateHash([](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getNumNeurons(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxConnections(); }, hash);
    updateHash([](const SynapseGroupInternal &g) { return g.getMaxSourceConnections(); }, hash);
    
    if(updateRole) {
        // Update hash with weight update model parameters and derived parameters
        updateHash([](const SynapseGroupInternal &g) { return g.getWUParams(); }, hash);
        updateHash([](const SynapseGroupInternal &g) { return g.getWUDerivedParams(); }, hash);

        // Update hash with presynaptic neuron population parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSrcNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getParams(); }, hash);
        
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSrcNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getSrcNeuronGroup()->getDerivedParams(); }, hash);

        // Update hash with postsynaptic neuron population parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isTrgNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getParams(); }, hash);
        
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isTrgNeuronParamReferenced, 
            [](const SynapseGroupInternal &g) { return g.getTrgNeuronGroup()->getDerivedParams(); }, hash);
    }


    // If we're updating a hash for a group with procedural connectivity or initialising connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) || (role == Role::ConnectivityInit)) {
        // Update hash with connectivity parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getParams(); }, hash);

        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getConnectivityInitialiser().getDerivedParams(); }, hash);
    }

    // If we're updating a hash for a group with Toeplitz connectivity
    if((getArchetype().getMatrixType() & SynapseMatrixConnectivity::TOEPLITZ) && updateRole) {
        // Update hash with connectivity parameters and derived parameters
        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isToeplitzConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getParams(); }, hash);

        updateParamHash<SynapseGroupMergedBase>(
            &SynapseGroupMergedBase::isToeplitzConnectivityInitParamReferenced,
            [](const SynapseGroupInternal &sg) { return sg.getToeplitzConnectivityInitialiser().getDerivedParams(); }, hash);
    }

    if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        // If this is an update role
        // **NOTE **global variable values aren't useful during initialization
        if(updateRole) {
            updateParamHash<SynapseGroupMergedBase>(
                &SynapseGroupMergedBase::isWUGlobalVarReferenced,
                [](const SynapseGroupInternal &sg) { return sg.getWUConstInitVals();  }, hash);
        }
    }
    // Otherwise (weights are individual or procedural)
    else {
        const bool connectInitRole = (role == Role::ConnectivityInit);
        const bool varInitRole = (role == Role::Init || role == Role::SparseInit);
        const bool proceduralWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::PROCEDURAL);
        const bool individualWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);
        const bool kernelWeights = (getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);

        // If synapse group has a kernel and we're either updating with procedural  
        // weights or initialising individual weights, update hash with kernel size
        if(!getArchetype().getKernelSize().empty() && 
            ((proceduralWeights && updateRole) || (connectInitRole && individualWeights) || (kernelWeights && !updateRole))) 
        {
            updateHash([](const SynapseGroupInternal &g) { return g.getKernelSize(); }, hash);
        }

        // If weights are procedural, we're initializing individual variables or we're initialising variables in a kernel
        // **NOTE** some of these won't actually be required - could do this per-variable in loop over vars
        if((proceduralWeights && updateRole) || (connectInitRole && !getArchetype().getKernelSize().empty())
           || (varInitRole && individualWeights) || (varInitRole && kernelWeights))
        {
            // Update hash with each group's variable initialisation parameters and derived parameters
            updateVarInitParamHash<SynapseGroupMergedBase, SynapseWUVarAdapter>(
                &SynapseGroupMergedBase::isWUVarInitParamReferenced, hash);
            updateVarInitDerivedParamHash<SynapseGroupMergedBase, SynapseWUVarAdapter>(
                &SynapseGroupMergedBase::isWUVarInitParamReferenced, hash);
        }
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
std::string SynapseGroupMergedBase::getVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication,
                                                const std::string &index, const std::string &prefix) const
{
    if (delay) {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return prefix + ((batchSize == 1) ? "DelaySlot" : "BatchDelaySlot");
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return prefix + "DelayOffset + " + index;
        }
        else {
            return prefix + "BatchDelayOffset + " + index;
        }
    }
    else {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return (batchSize == 1) ? "0" : "batch";
        }
        else if (varDuplication == VarAccessDuplication::SHARED || batchSize == 1) {
            return index;
        }
        else {
            return prefix + "BatchOffset + " + index;
        }
    }
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUParamReferenced(const std::string &paramName) const
{
    return isParamReferenced({getArchetypeCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUGlobalVarReferenced(const std::string &varName) const
{
    // If synapse group has global WU variables
    if(getArchetype().getMatrixType() & SynapseMatrixWeight::GLOBAL) {
        return isParamReferenced({getArchetypeCode()}, varName);
    }
    // Otherwise, return false
    else {
        return false;
    }
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isWUVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
{
    // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
    const auto *varInitSnippet = getArchetype().getWUVarInitialisers().at(varName).getSnippet();
    return isParamReferenced({varInitSnippet->getCode()}, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSparseConnectivityInitParamReferenced(const std::string &paramName) const
{
    const auto *snippet = getArchetype().getConnectivityInitialiser().getSnippet();
    const auto rowBuildStateVars = snippet->getRowBuildStateVars();
    const auto colBuildStateVars = snippet->getColBuildStateVars();

    // Build list of code strings containing row build code and any row build state variable values
    std::vector<std::string> codeStrings{snippet->getRowBuildCode(), snippet->getColBuildCode()};
    std::transform(rowBuildStateVars.cbegin(), rowBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });
    std::transform(colBuildStateVars.cbegin(), colBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });

    return isParamReferenced(codeStrings, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isToeplitzConnectivityInitParamReferenced(const std::string &paramName) const
{
    const auto *snippet = getArchetype().getToeplitzConnectivityInitialiser().getSnippet();
    const auto diagonalBuildStateVars = snippet->getDiagonalBuildStateVars();

    // Build list of code strings containing diagonal build code and any diagonal build state variable values
    std::vector<std::string> codeStrings{snippet->getDiagonalBuildCode()};
    std::transform(diagonalBuildStateVars.cbegin(), diagonalBuildStateVars.cend(), std::back_inserter(codeStrings),
                   [](const Snippet::Base::ParamVal &p) { return p.value; });
   
    return isParamReferenced(codeStrings, paramName);
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isSrcNeuronParamReferenced(const std::string &paramName) const
{
    return isParamReferenced({getArchetypeCode()}, paramName + "_pre");
}
//----------------------------------------------------------------------------
bool SynapseGroupMergedBase::isTrgNeuronParamReferenced(const std::string &paramName) const
{
    return isParamReferenced({getArchetypeCode()}, paramName +  "_post");
}
