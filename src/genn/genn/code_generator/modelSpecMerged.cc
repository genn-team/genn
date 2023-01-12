#include "code_generator/modelSpecMerged.h"

// GeNN includes
#include "logging.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

using namespace CodeGenerator;

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
// CodeGenerator::ModelSpecMerged
//----------------------------------------------------------------------------
ModelSpecMerged::ModelSpecMerged(const ModelSpecInternal &model, const BackendBase &backend)
:   m_Model(model), m_NeuronUpdateSupportCode("NeuronUpdateSupportCode"), m_PostsynapticDynamicsSupportCode("PostsynapticDynamicsSupportCode"),
    m_PresynapticUpdateSupportCode("PresynapticUpdateSupportCode"), m_PostsynapticUpdateSupportCode("PostsynapticUpdateSupportCode"),
    m_SynapseDynamicsSupportCode("SynapseDynamicsSupportCode")
{
    LOGD_CODE_GEN << "Merging neuron update groups:";
    createMergedGroupsHash(model, backend, model.getNeuronGroups(), m_MergedNeuronUpdateGroups,
                           [](const NeuronGroupInternal &){ return true; },
                           &NeuronGroupInternal::getHashDigest);

    LOGD_CODE_GEN << "Merging presynaptic update groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedPresynapticUpdateGroups,
                           [](const SynapseGroupInternal &sg) { return (sg.isSpikeEventRequired() || sg.isTrueSpikeRequired()); },
                           &SynapseGroupInternal::getWUHashDigest);

    LOGD_CODE_GEN << "Merging postsynaptic update groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedPostsynapticUpdateGroups,
                           [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getLearnPostCode().empty(); },
                           &SynapseGroupInternal::getWUHashDigest);

    LOGD_CODE_GEN << "Merging synapse dynamics update groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseDynamicsGroups,
                           [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
                           &SynapseGroupInternal::getWUHashDigest);

    LOGD_CODE_GEN << "Merging neuron initialization groups:";
    createMergedGroupsHash(model, backend, model.getNeuronGroups(), m_MergedNeuronInitGroups,
                           [](const NeuronGroupInternal &){ return true; },
                           &NeuronGroupInternal::getInitHashDigest);
                           
    LOGD_CODE_GEN << "Merging synapse initialization groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseInitGroups,
                           [](const SynapseGroupInternal &sg)
                           {
                               return (((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE)
                                        || (sg.getMatrixType() & SynapseMatrixWeight::KERNEL))
                                        && sg.isWUVarInitRequired());
                           },
                           &SynapseGroupInternal::getWUInitHashDigest);

    LOGD_CODE_GEN << "Merging custom update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomUpdates(), m_MergedCustomUpdateInitGroups,
                           [](const CustomUpdateInternal &cg) { return cg.isVarInitRequired(); },
                           &CustomUpdateInternal::getInitHashDigest);

    LOGD_CODE_GEN << "Merging custom weight update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomWUUpdateInitGroups,
                           [](const CustomUpdateWUInternal &cg) 
                           {
                               return (((cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE)
                                        || (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL))
                                        && cg.isVarInitRequired());
                           },
                           &CustomUpdateWUInternal::getInitHashDigest);

    LOGD_CODE_GEN << "Merging synapse connectivity initialisation groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseConnectivityInitGroups,
                           [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
                           &SynapseGroupInternal::getConnectivityInitHashDigest);

    LOGD_CODE_GEN << "Merging synapse sparse initialization groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseSparseInitGroups,
                           [&backend](const SynapseGroupInternal &sg)
                           {
                               return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && 
                                       (sg.isWUVarInitRequired()
                                        || (backend.isPostsynapticRemapRequired() && !sg.getWUModel()->getLearnPostCode().empty())));
                           },
                           &SynapseGroupInternal::getWUInitHashDigest);

    LOGD_CODE_GEN << "Merging custom sparse weight update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomWUUpdateSparseInitGroups,
                           [](const CustomUpdateWUInternal &cg) 
                           {
                               return (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) && cg.isVarInitRequired(); 
                           },
                           &CustomUpdateWUInternal::getInitHashDigest);
    
    LOGD_CODE_GEN << "Merging custom connectivity update presynaptic initialisation groups:";
    createMergedGroupsHash(model, backend, model.getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdatePreInitGroups,
                           [&backend](const CustomConnectivityUpdateInternal &cg) 
                           {
                               return (cg.isPreVarInitRequired() || (backend.isPopulationRNGInitialisedOnDevice() && cg.isRowSimRNGRequired()));     
                           },
                           &CustomConnectivityUpdateInternal::getInitHashDigest);

    LOGD_CODE_GEN << "Merging custom connectivity update postsynaptic initialisation groups:";
    createMergedGroupsHash(model, backend, model.getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdatePostInitGroups,
                           [](const CustomConnectivityUpdateInternal &cg) { return cg.isPostVarInitRequired(); },
                           &CustomConnectivityUpdateInternal::getInitHashDigest);

    LOGD_CODE_GEN << "Merging custom connectivity update sparse initialisation groups:";
    createMergedGroupsHash(model, backend, model.getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdateSparseInitGroups,
                           [](const CustomConnectivityUpdateInternal &cg) { return cg.isVarInitRequired(); },
                           &CustomConnectivityUpdateInternal::getInitHashDigest);

    LOGD_CODE_GEN << "Merging neuron groups which require their spike queues updating:";
    createMergedGroupsHash(model, backend, model.getNeuronGroups(), m_MergedNeuronSpikeQueueUpdateGroups,
                           [](const NeuronGroupInternal &){ return true; },
                           &NeuronGroupInternal::getSpikeQueueUpdateHashDigest);

    LOGD_CODE_GEN << "Merging neuron groups which require their previous spike times updating:";
    createMergedGroupsHash(model, backend, model.getNeuronGroups(), m_MergedNeuronPrevSpikeTimeUpdateGroups,
                           [](const NeuronGroupInternal &ng){ return (ng.isPrevSpikeTimeRequired() || ng.isPrevSpikeEventTimeRequired()); },
                           &NeuronGroupInternal::getPrevSpikeTimeUpdateHashDigest);

    // Build vector of merged synapse groups which require dendritic delay
    std::vector<std::reference_wrapper<const SynapseGroupInternal>> synapseGroupsWithDendriticDelay;
    for(const auto &n : model.getNeuronGroups()) {
        for(const auto *sg : n.second.getFusedPSMInSyn()) {
            if(sg->isDendriticDelayRequired()) {
                synapseGroupsWithDendriticDelay.push_back(std::cref(*sg));
            }
        }
    }
    LOGD_CODE_GEN << "Merging synapse groups which require their dendritic delay updating:";
    createMergedGroupsHash(model, backend, synapseGroupsWithDendriticDelay, m_MergedSynapseDendriticDelayUpdateGroups,
                           &SynapseGroupInternal::getDendriticDelayUpdateHashDigest);

    LOGD_CODE_GEN << "Merging synapse groups which require host code to initialise their synaptic connectivity:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseConnectivityHostInitGroups,
                           [](const SynapseGroupInternal &sg)
                           { 
                               return !sg.getConnectivityInitialiser().getSnippet()->getHostInitCode().empty();
                           },
                           &SynapseGroupInternal::getConnectivityHostInitHashDigest, true);

    LOGD_CODE_GEN << "Merging custom update groups:";
    createMergedGroupsHash(model, backend, model.getCustomUpdates(), m_MergedCustomUpdateGroups,
                           [](const CustomUpdateInternal &) { return true; },
                           &CustomUpdateInternal::getHashDigest);

    LOGD_CODE_GEN << "Merging custom weight update groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomUpdateWUGroups,
                           [](const CustomUpdateWUInternal &cg) { return !cg.isTransposeOperation(); },
                           &CustomUpdateWUInternal::getHashDigest);

    LOGD_CODE_GEN << "Merging custom weight transpose update groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomUpdateTransposeWUGroups,
                           [](const CustomUpdateWUInternal &cg) { return cg.isTransposeOperation(); },
                           &CustomUpdateWUInternal::getHashDigest);

    if(backend.isHostReductionRequired()) {
        LOGD_CODE_GEN << "Merging custom weight update groups:";
        createMergedGroupsHash(model, backend, model.getCustomUpdates(), m_MergedCustomUpdateHostReductionGroups,
                               [](const CustomUpdateInternal &cg) { return cg.isBatchReduction(); },
                               &CustomUpdateInternal::getHashDigest, true);

        LOGD_CODE_GEN << "Merging custom weight transpose update groups:";
        createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomWUUpdateHostReductionGroups,
                               [](const CustomUpdateWUInternal &cg) { return cg.isBatchReduction(); },
                               &CustomUpdateWUInternal::getHashDigest, true);
    }

    LOGD_CODE_GEN << "Merging custom connectivity update groups:";
    createMergedGroupsHash(model, backend, model.getCustomConnectivityUpdates(), m_MergedCustomConnectivityUpdateGroups,
                           [](const CustomConnectivityUpdateInternal &cg) { return !cg.getCustomConnectivityUpdateModel()->getRowUpdateCode().empty(); },
                           &CustomConnectivityUpdateInternal::getHashDigest);

    LOGD_CODE_GEN << "Merging custom connectivity host update groups:";
    createMergedGroupsHash(model, backend, model.getCustomConnectivityUpdates(), m_MergedCustomConnectivityHostUpdateGroups,
                           [](const CustomConnectivityUpdateInternal &cg) { return !cg.getCustomConnectivityUpdateModel()->getHostUpdateCode().empty(); },
                           &CustomConnectivityUpdateInternal::getHashDigest, true);

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
        if(std::any_of(e.second.cbegin(), e.second.cend(),
                       [](const MergedEGPDestinations::value_type &g) 
                       {
                           return Utils::isTypePointer(g.second.type); 
                       }))
        {
            return true;
        }
    }

    return false;
}
