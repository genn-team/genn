#include "code_generator/modelSpecMerged.h"

// GeNN includes
#include "logging.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

using namespace CodeGenerator;

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

    LOGD_CODE_GEN << "Merging custom update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomUpdates(), m_MergedCustomUpdateInitGroups,
                           [](const CustomUpdateInternal &cg) { return cg.isVarInitRequired(); },
                           &CustomUpdateInternal::getInitHashDigest);

    LOGD_CODE_GEN << "Merging custom dense weight update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomWUUpdateDenseInitGroups,
                           [](const CustomUpdateWUInternal &cg) { return (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE) && cg.isVarInitRequired(); },
                           &CustomUpdateWUInternal::getInitHashDigest);

    LOGD_CODE_GEN << "Merging synapse dense initialization groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseDenseInitGroups,
                           [](const SynapseGroupInternal &sg)
                           {
                               return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && sg.isWUVarInitRequired());
                           },
                           &SynapseGroupInternal::getWUInitHashDigest);

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
                                        || backend.isSynRemapRequired(sg)
                                        || (backend.isPostsynapticRemapRequired() && !sg.getWUModel()->getLearnPostCode().empty())));
                           },
                           &SynapseGroupInternal::getWUInitHashDigest);

    LOGD_CODE_GEN << "Merging custom sparse weight update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomWUUpdateSparseInitGroups,
                           [](const CustomUpdateWUInternal &cg) { return (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) && cg.isVarInitRequired(); },
                           &CustomUpdateWUInternal::getInitHashDigest);

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
        for(const auto *sg : n.second.getMergedInSyn()) {
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
                               return (!sg.isWeightSharingSlave() && !sg.getConnectivityInitialiser().getSnippet()->getHostInitCode().empty()); 
                           },
                           &SynapseGroupInternal::getConnectivityHostInitHashDigest);

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

    // Loop through merged neuron groups
    for(const auto &ng : m_MergedNeuronUpdateGroups) {
        // Add neuron support code
        m_NeuronUpdateSupportCode.addSupportCode(ng.getArchetype().getNeuronModel()->getSupportCode());

        // Loop through merged postsynaptic models and add their support code
        for(const auto &sg : ng.getArchetype().getMergedInSyn()) {
            m_PostsynapticDynamicsSupportCode.addSupportCode(sg->getPSModel()->getSupportCode());
        }
    }

    // Loop through merged presynaptic update groups and add support code
    for(const auto &sg : m_MergedPresynapticUpdateGroups) {
        m_PresynapticUpdateSupportCode.addSupportCode(sg.getArchetype().getWUModel()->getSimSupportCode());
    }

    // Loop through merged postsynaptic update groups and add support code
    for(const auto &sg : m_MergedPostsynapticUpdateGroups) {
        m_PostsynapticUpdateSupportCode.addSupportCode(sg.getArchetype().getWUModel()->getLearnPostSupportCode());
    }

    // Loop through merged synapse dynamics groups and add support code
    for(const auto &sg : m_MergedSynapseDynamicsGroups) {
        m_SynapseDynamicsSupportCode.addSupportCode(sg.getArchetype().getWUModel()->getSynapseDynamicsSuppportCode());
    }
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getNeuronUpdateModuleHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Concatenate hash digest of model properties
    Utils::updateHash(getModel().getHashDigest(false), hash);

    // Concatenate hash digest of neuron update groups
    for(const auto &g : m_MergedNeuronUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // **NOTE** all properties of neuron spike queue and previous spike time
    // updates are also included in neuron update groups so no need to hash
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getSynapseUpdateModuleHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Concatenate hash of model properties
    Utils::updateHash(getModel().getHashDigest(false), hash);

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

    // **NOTE** all properties of synapse dendritic delay updates
    //  are also included in synapse update groups so no need to hash
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getCustomUpdateModuleHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Concatenate hash of model properties
    Utils::updateHash(getModel().getHashDigest(false), hash);
    
    // Concatenate hash digest of custom update groups
    for(const auto &g : m_MergedCustomUpdateGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom WU update groups
    for(const auto &g : m_MergedCustomUpdateWUGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom transpose WU update groups
    for(const auto &g : m_MergedCustomUpdateTransposeWUGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type ModelSpecMerged::getInitModuleHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Concatenate hash of model properties
    // **NOTE** RNG seed effects initialisation model
    Utils::updateHash(getModel().getHashDigest(true), hash);
    
    // Concatenate hash digest of neuron init groups
    for(const auto &g : m_MergedNeuronInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of synapse dense init groups
    for(const auto &g : m_MergedSynapseDenseInitGroups) {
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

    // Concatenate hash digest of custom dense WU update init groups
    for(const auto &g : m_MergedCustomWUUpdateDenseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
    }

    // Concatenate hash digest of custom sparse WU update init groups
    for(const auto &g : m_MergedCustomWUUpdateSparseInitGroups) {
        Utils::updateHash(g.getHashDigest(), hash);
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
