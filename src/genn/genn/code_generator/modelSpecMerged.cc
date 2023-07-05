#include "code_generator/modelSpecMerged.h"

// GeNN includes
#include "gennUtils.h"
#include "logging.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
template<typename T>
void assignGroups(const BackendBase &backend, std::vector<T> &groups, BackendBase::MemorySpaces &memorySpaces)
{
    // Loop through groups and assign groups
    for(auto &g : groups) {
        g.assignMemorySpaces(backend, memorySpaces);
    }
}
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::ModelSpecMerged
//----------------------------------------------------------------------------
ModelSpecMerged::ModelSpecMerged(const ModelSpecInternal &model, const BackendBase &backend)
:   m_Model(model), m_NeuronUpdateSupportCode("NeuronUpdateSupportCode"), m_PostsynapticDynamicsSupportCode("PostsynapticDynamicsSupportCode"),
    m_PresynapticUpdateSupportCode("PresynapticUpdateSupportCode"), m_PostsynapticUpdateSupportCode("PostsynapticUpdateSupportCode"),
    m_SynapseDynamicsSupportCode("SynapseDynamicsSupportCode"), m_TypeContext{{"scalar", model.getPrecision()}, {"timepoint", model.getTimePrecision()}}
{
  
    // Get memory spaces available to this backend
    // **NOTE** Memory spaces are given out on a first-come, first-serve basis so subsequent groups are in preferential order
    auto memorySpaces = backend.getMergedGroupMemorySpaces(*this);

    // Loop through dendritic delay update groups and assign memory spaces
    assignGroups(backend, m_MergedSynapseDendriticDelayUpdateGroups, memorySpaces);

    // Loop through merged presynaptic update groups, assign memory spaces and add support code
    for(auto &sg : m_MergedPresynapticUpdateGroups) {
        sg.assignMemorySpaces(backend, memorySpaces);
        m_PresynapticUpdateSupportCode.addSupportCode(sg.getArchetype().getWUModel()->getSimSupportCode());
    }

    // Loop through merged postsynaptic update groups, assign memory spaces and add support code
    for(auto &sg : m_MergedPostsynapticUpdateGroups) {
        sg.assignMemorySpaces(backend, memorySpaces);
        m_PostsynapticUpdateSupportCode.addSupportCode(sg.getArchetype().getWUModel()->getLearnPostSupportCode());
    }

    // Loop through merged synapse dynamics groups, assign memory spaces and add support code
    for(auto &sg : m_MergedSynapseDynamicsGroups) {
        sg.assignMemorySpaces(backend, memorySpaces);
        m_SynapseDynamicsSupportCode.addSupportCode(sg.getArchetype().getWUModel()->getSynapseDynamicsSuppportCode());
    }

    // Loop through previous spike time and spike queue update groups and assign memory spaces
    assignGroups(backend, m_MergedNeuronPrevSpikeTimeUpdateGroups, memorySpaces);
    assignGroups(backend, m_MergedNeuronSpikeQueueUpdateGroups, memorySpaces);
    
    // Loop through merged neuron groups
    for(auto &ng : m_MergedNeuronUpdateGroups) {
        // Assign memory spaces
        ng.assignMemorySpaces(backend, memorySpaces);

        // Add neuron support code
        m_NeuronUpdateSupportCode.addSupportCode(ng.getArchetype().getNeuronModel()->getSupportCode());

        // Loop through merged postsynaptic models and add their support code
        for(const auto &sg : ng.getArchetype().getFusedPSMInSyn()) {
            m_PostsynapticDynamicsSupportCode.addSupportCode(sg->getPSModel()->getSupportCode());
        }
    }

    // Loop through custom update groups and assign memory spaces
    assignGroups(backend, m_MergedCustomUpdateGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomUpdateWUGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomUpdateTransposeWUGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomConnectivityUpdateGroups, memorySpaces);

    // Loop through init groups and assign memory spaces
    assignGroups(backend, m_MergedNeuronInitGroups, memorySpaces);
    assignGroups(backend, m_MergedSynapseInitGroups, memorySpaces);
    assignGroups(backend, m_MergedSynapseSparseInitGroups, memorySpaces);
    assignGroups(backend, m_MergedSynapseConnectivityInitGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomUpdateInitGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomWUUpdateInitGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomWUUpdateSparseInitGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomConnectivityUpdatePreInitGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomConnectivityUpdatePostInitGroups, memorySpaces);
    assignGroups(backend, m_MergedCustomConnectivityUpdateSparseInitGroups, memorySpaces);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedNeuronUpdateGroups(const BackendBase &backend, GenMergedGroupFn<NeuronUpdateGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getNeuronGroups(), m_MergedNeuronUpdateGroups,
                        [](const NeuronGroupInternal &){ return true; },
                        &NeuronGroupInternal::getHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedPresynapticUpdateGroups(const BackendBase &backend, GenMergedGroupFn<PresynapticUpdateGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedPresynapticUpdateGroups,
                        [](const SynapseGroupInternal &sg) { return (sg.isSpikeEventRequired() || sg.isTrueSpikeRequired()); },
                        &SynapseGroupInternal::getWUHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedPostsynapticUpdateGroups(const BackendBase &backend, GenMergedGroupFn<PostsynapticUpdateGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedPostsynapticUpdateGroups,
                        [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getLearnPostCode().empty(); },
                        &SynapseGroupInternal::getWUHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseDynamicsGroups(const BackendBase &backend, GenMergedGroupFn<SynapseDynamicsGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseDynamicsGroups,
                        [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
                        &SynapseGroupInternal::getWUHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateGroups(const BackendBase &backend, const std::string &updateGroupName, 
                                    GenMergedGroupFn<CustomUpdateGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomUpdates(), m_MergedCustomUpdateGroups,
                        [&updateGroupName](const CustomUpdateInternal &cg) { return cg.getUpdateGroupName() == updateGroupName; },
                        &CustomUpdateInternal::getHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateWUGroups(const BackendBase &backend, const std::string &updateGroupName, 
                                    GenMergedGroupFn<CustomUpdateWUGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomUpdateWUGroups,
                        [&updateGroupName](const CustomUpdateWUInternal &cg) 
                        {
                            return (!cg.isTransposeOperation() && cg.getUpdateGroupName() == updateGroupName); 
                        },
                        &CustomUpdateWUInternal::getHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateTransposeWUGroups(const BackendBase &backend, const std::string &updateGroupName, 
                                            GenMergedGroupFn<CustomUpdateTransposeWUGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomUpdateTransposeWUGroups,
                        [&updateGroupName](const CustomUpdateWUInternal &cg)
                        {
                            return (cg.isTransposeOperation() && cg.getUpdateGroupName() == updateGroupName); 
                        },
                        &CustomUpdateWUInternal::getHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateHostReductionGroups(const BackendBase &backend, const std::string &updateGroupName, 
                                                GenMergedGroupFn<CustomUpdateHostReductionGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomUpdates(), m_MergedCustomUpdateHostReductionGroups,
                        [&updateGroupName](const CustomUpdateInternal &cg)
                        {
                            return (cg.isBatchReduction() && cg.getUpdateGroupName() == updateGroupName); 
                        },
                        &CustomUpdateInternal::getHashDigest, generateGroup, true);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomWUUpdateHostReductionGroups(const BackendBase &backend, const std::string &updateGroupName, 
                                                GenMergedGroupFn<CustomWUUpdateHostReductionGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateHostReductionGroups,
                        [&updateGroupName](const CustomUpdateWUInternal &cg)
                        {
                            return (cg.isBatchReduction() && cg.getUpdateGroupName() == updateGroupName); 
                        },
                        &CustomUpdateWUInternal::getHashDigest, generateGroup, true);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityUpdateGroups(const BackendBase &backend, const std::string &updateGroupName, 
                                                GenMergedGroupFn<CustomConnectivityUpdateGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdateGroups,
                        [&updateGroupName](const CustomConnectivityUpdateInternal &cg)
                        {
                            return (!cg.getCustomConnectivityUpdateModel()->getRowUpdateCode().empty() && cg.getUpdateGroupName() == updateGroupName); 
                        },
                        &CustomConnectivityUpdateInternal::getHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityHostUpdateGroups(BackendBase &backend, const std::string &updateGroupName, 
                                                    GenMergedGroupFn<CustomConnectivityHostUpdateGroupMerged> generateGroup)
{
        createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityHostUpdateGroups,
                        [&updateGroupName](const CustomConnectivityUpdateInternal &cg) 
                        { 
                            return (!cg.getCustomConnectivityUpdateModel()->getHostUpdateCode().empty() && cg.getUpdateGroupName() == updateGroupName); 
                        },
                        &CustomConnectivityUpdateInternal::getHashDigest, generateGroup, true);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedNeuronSpikeQueueUpdateGroups(const BackendBase &backend, GenMergedGroupFn<NeuronSpikeQueueUpdateGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getNeuronGroups(), m_MergedNeuronSpikeQueueUpdateGroups,
                        [](const NeuronGroupInternal &){ return true; },
                        &NeuronGroupInternal::getSpikeQueueUpdateHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedNeuronPrevSpikeTimeUpdateGroups(const BackendBase &backend, GenMergedGroupFn<NeuronPrevSpikeTimeUpdateGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getNeuronGroups(), m_MergedNeuronPrevSpikeTimeUpdateGroups,
                        [](const NeuronGroupInternal &ng){ return (ng.isPrevSpikeTimeRequired() || ng.isPrevSpikeEventTimeRequired()); },
                        &NeuronGroupInternal::getPrevSpikeTimeUpdateHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseDendriticDelayUpdateGroups(const BackendBase &backend, GenMergedGroupFn<SynapseDendriticDelayUpdateGroupMerged> generateGroup)
{
    std::vector<std::reference_wrapper<const SynapseGroupInternal>> synapseGroupsWithDendriticDelay;
    for(const auto &n : getModel().getNeuronGroups()) {
        for(const auto *sg : n.second.getFusedPSMInSyn()) {
            if(sg->isDendriticDelayRequired()) {
                synapseGroupsWithDendriticDelay.push_back(std::cref(*sg));
            }
        }
    }
    createMergedGroups(backend, synapseGroupsWithDendriticDelay, m_MergedSynapseDendriticDelayUpdateGroups,
                        &SynapseGroupInternal::getDendriticDelayUpdateHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedNeuronInitGroups(const BackendBase &backend, GenMergedGroupFn<NeuronInitGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getNeuronGroups(), m_MergedNeuronInitGroups,
                        [](const NeuronGroupInternal &){ return true; },
                        &NeuronGroupInternal::getInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomUpdateInitGroups(const BackendBase &backend, GenMergedGroupFn<CustomUpdateInitGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomUpdates(), m_MergedCustomUpdateInitGroups,
                        [](const CustomUpdateInternal &cg) { return cg.isVarInitRequired(); },
                        &CustomUpdateInternal::getInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomWUUpdateInitGroups(const BackendBase &backend, GenMergedGroupFn<CustomWUUpdateInitGroupMerged> generateGroup)
{
        createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateInitGroups,
                        [](const CustomUpdateWUInternal &cg) 
                        {
                            return (((cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE)
                                        || (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL))
                                        && cg.isVarInitRequired());
                        },
                        &CustomUpdateWUInternal::getInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseInitGroups(const BackendBase &backend, GenMergedGroupFn<SynapseInitGroupMerged> generateGroup)
{
        createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseInitGroups,
                        [](const SynapseGroupInternal &sg)
                        {
                            return (((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)
                                        || (sg.getMatrixType() & SynapseMatrixWeight::KERNEL))
                                        && sg.isWUVarInitRequired());
                        },
                        &SynapseGroupInternal::getWUInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseConnectivityInitGroups(const BackendBase &backend, GenMergedGroupFn<SynapseConnectivityInitGroupMerged> generateGroup)
{
        createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseConnectivityInitGroups,
                        [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
                        &SynapseGroupInternal::getConnectivityInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseSparseInitGroups(const BackendBase &backend, GenMergedGroupFn<SynapseSparseInitGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseSparseInitGroups,
                        [&backend](const SynapseGroupInternal &sg)
                        {
                            return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && 
                                    (sg.isWUVarInitRequired()
                                    || (backend.isPostsynapticRemapRequired() && !sg.getWUModel()->getLearnPostCode().empty())));
                        },
                        &SynapseGroupInternal::getWUInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomWUUpdateSparseInitGroups(const BackendBase &backend, GenMergedGroupFn<CustomWUUpdateSparseInitGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomWUUpdates(), m_MergedCustomWUUpdateSparseInitGroups,
                        [](const CustomUpdateWUInternal &cg) 
                        {
                            return (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) && cg.isVarInitRequired(); 
                        },
                        &CustomUpdateWUInternal::getInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityUpdatePreInitGroups(const BackendBase &backend, GenMergedGroupFn<CustomConnectivityUpdatePreInitGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdatePreInitGroups,
                        [&backend](const CustomConnectivityUpdateInternal &cg) 
                        {
                            return (cg.isPreVarInitRequired() || (backend.isPopulationRNGInitialisedOnDevice() && Utils::isRNGRequired(cg.getRowUpdateCodeTokens())));     
                        },
                        &CustomConnectivityUpdateInternal::getInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityUpdatePostInitGroups(const BackendBase &backend, GenMergedGroupFn<CustomConnectivityUpdatePostInitGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdatePostInitGroups,
                        [](const CustomConnectivityUpdateInternal &cg) { return cg.isPostVarInitRequired(); },
                        &CustomConnectivityUpdateInternal::getInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedCustomConnectivityUpdateSparseInitGroups(const BackendBase &backend, GenMergedGroupFn<CustomConnectivityUpdateSparseInitGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdateSparseInitGroups,
                        [](const CustomConnectivityUpdateInternal &cg) { return cg.isVarInitRequired(); },
                        &CustomConnectivityUpdateInternal::getInitHashDigest, generateGroup);
}
//----------------------------------------------------------------------------
void ModelSpecMerged::genMergedSynapseConnectivityHostInitGroups(const BackendBase &backend, GenMergedGroupFn<SynapseConnectivityHostInitGroupMerged> generateGroup)
{
    createMergedGroups(backend, getModel().getSynapseGroups(), m_MergedSynapseConnectivityHostInitGroups,
                        [](const SynapseGroupInternal &sg)
                        { 
                            return !sg.getConnectivityInitialiser().getSnippet()->getHostInitCode().empty();
                        },
                        &SynapseGroupInternal::getConnectivityHostInitHashDigest, generateGroup, true);
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getHashDigest(const BackendBase &backend) const
{
    boost::uuids::detail::sha1 hash;

    // Concatenate hash digest of model properties
    Utils::updateHash(getModel().getHashDigest(), hash);

    // Concatenate hash digest of backend properties
    Utils::updateHash(backend.getHashDigest(), hash);

    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of neuron update groups
    for(const auto &g : m_MergedNeuronUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of presynaptic update groups
    for(const auto &g : m_MergedPresynapticUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of postsynaptic update groups
    for(const auto &g : m_MergedPostsynapticUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of synapse dynamics groups
    for(const auto &g : m_MergedSynapseDynamicsGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom update groups
    for(const auto &g : m_MergedCustomUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom WU update groups
    for(const auto &g : m_MergedCustomUpdateWUGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom connectivity update groups
    for(const auto &g : m_MergedCustomConnectivityUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom transpose WU update groups
    for(const auto &g : m_MergedCustomUpdateTransposeWUGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of neuron init groups
    for(const auto &g : m_MergedNeuronInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of synapse init groups
    for(const auto &g : m_MergedSynapseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Update hash with hash digest of synapse sparse init groups
    for(const auto &g : m_MergedSynapseSparseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of synapse connectivity init groups
    for(const auto &g : m_MergedSynapseConnectivityInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom update init groups
    for(const auto &g : m_MergedCustomUpdateInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom WU update init groups
    for(const auto &g : m_MergedCustomWUUpdateInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom sparse WU update init groups
    for(const auto &g : m_MergedCustomWUUpdateSparseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity pre update init group
    for (const auto &g : m_MergedCustomConnectivityUpdatePreInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity post update init group
    for (const auto &g : m_MergedCustomConnectivityUpdatePostInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity sparse update init group
    for (const auto &g : m_MergedCustomConnectivityUpdateSparseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Update hash with each group's variable locations
    // **NOTE** these only effects the runner - doesn't matter for modules so this is done he
    for(const auto &g : getModel().getNeuronGroups()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    for(const auto &g : getModel().getSynapseGroups()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    for(const auto &g : getModel().getLocalCurrentSources()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }
    
    for(const auto &g : getModel().getCustomUpdates()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    for(const auto &g : getModel().getCustomWUUpdates()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    for(const auto &g : getModel().getCustomConnectivityUpdates()) {
        Utils::updateHash(g.second.getName(), hash);
        Utils::updateHash(g.second.getSynapseGroup()->getName(), hash);
        Utils::updateHash(g.second.getVarLocationHashDigest(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getNeuronUpdateArchetypeHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Add hash of model batch size
    Utils::updateHash(getModel().getBatchSize(), hash);

    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of archetype neuron update group
    for(const auto &g : m_MergedNeuronUpdateGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    // **NOTE** all properties of neuron spike queue and previous spike time
    // updates are also included in neuron update groups so no need to hash
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getSynapseUpdateArchetypeHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Add hash of model batch size
    Utils::updateHash(getModel().getBatchSize(), hash);

    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of archetype presynaptic update group
    for(const auto &g : m_MergedPresynapticUpdateGroups) {
        Utils::updateHash(g.getArchetype().getWUHashDigest(), hash);
    }

    // Concatenate hash digest of archetype postsynaptic update group
    for(const auto &g : m_MergedPostsynapticUpdateGroups) {
        Utils::updateHash(g.getArchetype().getWUHashDigest(), hash);
    }

    // Concatenate hash digest of archetype synapse dynamics group
    for(const auto &g : m_MergedSynapseDynamicsGroups) {
        Utils::updateHash(g.getArchetype().getWUHashDigest(), hash);
    }

    // **NOTE** all properties of synapse dendritic delay updates
    //  are also included in synapse update groups so no need to hash

    return hash.get_digest();
    
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getCustomUpdateArchetypeHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Add hash of model batch size
    Utils::updateHash(getModel().getBatchSize(), hash);
    
    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of archetype custom update group
    for(const auto &g : m_MergedCustomUpdateGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom WU update group
    for(const auto &g : m_MergedCustomUpdateWUGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom transpose WU update group
    for(const auto &g : m_MergedCustomUpdateTransposeWUGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivityupdate group
    for(const auto &g : m_MergedCustomConnectivityUpdateGroups) {
        Utils::updateHash(g.getArchetype().getHashDigest(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getInitArchetypeHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Add hash of model batch size
    Utils::updateHash(getModel().getBatchSize(), hash);
    
    // Concatenate hash digest of GeNN version
    Utils::updateHash(GENN_VERSION, hash);

    // Concatenate hash digest of git hash
    // **NOTE** it would be nicer to actually treat git hash as a hash but not really important
    Utils::updateHash(GIT_HASH, hash);

    // Concatenate hash digest of archetype neuron init group
    for(const auto &g : m_MergedNeuronInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype synapse dense init group
    for(const auto &g : m_MergedSynapseInitGroups) {
        Utils::updateHash(g.getArchetype().getWUInitHashDigest(), hash);
    }

    // Update hash with hash digest of archetype synapse sparse init group
    for(const auto &g : m_MergedSynapseSparseInitGroups) {
        Utils::updateHash(g.getArchetype().getWUInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype synapse connectivity init group
    for(const auto &g : m_MergedSynapseConnectivityInitGroups) {
        Utils::updateHash(g.getArchetype().getConnectivityInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom update init group
    for(const auto &g : m_MergedCustomUpdateInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom WU update init group
    for(const auto &g : m_MergedCustomWUUpdateInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom sparse WU update init group
    for(const auto &g : m_MergedCustomWUUpdateSparseInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity pre update init group
    for (const auto &g : m_MergedCustomConnectivityUpdatePreInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity post update init group
    for (const auto &g : m_MergedCustomConnectivityUpdatePostInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    // Concatenate hash digest of archetype custom connectivity sparse update init group
    for (const auto &g : m_MergedCustomConnectivityUpdateSparseInitGroups) {
        Utils::updateHash(g.getArchetype().getInitHashDigest(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
bool ModelSpecMerged::anyPointerEGPs() const
{
    // Loop through grouped merged EGPs
    for(const auto &e : m_MergedEGPs) {
        // If there's any pointer EGPs, return true
        // **TODO** without scalar EGPS, all EGPS are pointer EGPS!
        if(std::any_of(e.second.cbegin(), e.second.cend(),
                       [](const MergedEGPDestinations::value_type &g){ return g.second.type.isPointer(); }))
        {
            return true;
        }
    }

    return false;
}
