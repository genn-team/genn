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
    createMergedGroups(model, backend, model.getNeuronGroups(), m_MergedNeuronUpdateGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b){ return a.canBeMerged(b); });

    LOGD_CODE_GEN << "Merging presynaptic update groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedPresynapticUpdateGroups,
                           [](const SynapseGroupInternal &sg) { return (sg.isSpikeEventRequired() || sg.isTrueSpikeRequired()); },
                           &SynapseGroupInternal::updateWUHash);

    LOGD_CODE_GEN << "Merging postsynaptic update groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedPostsynapticUpdateGroups,
                           [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getLearnPostCode().empty(); },
                           &SynapseGroupInternal::updateWUHash);

    LOGD_CODE_GEN << "Merging synapse dynamics update groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseDynamicsGroups,
                           [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
                           &SynapseGroupInternal::updateWUHash);

    LOGD_CODE_GEN << "Merging neuron initialization groups:";
    createMergedGroups(model, backend, model.getNeuronGroups(), m_MergedNeuronInitGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b){ return a.canInitBeMerged(b); });

    LOGD_CODE_GEN << "Merging custom update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomUpdates(), m_MergedCustomUpdateInitGroups,
                           [](const CustomUpdateInternal &cg) { return cg.isVarInitRequired(); },
                           &CustomUpdateInternal::updateInitHash);

    LOGD_CODE_GEN << "Merging custom dense weight update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomWUUpdateDenseInitGroups,
                           [](const CustomUpdateWUInternal &cg) { return (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::DENSE) && cg.isVarInitRequired(); },
                           &CustomUpdateWUInternal::updateInitHash);

    LOGD_CODE_GEN << "Merging synapse dense initialization groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseDenseInitGroups,
                           [](const SynapseGroupInternal &sg)
                           {
                               return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && sg.isWUVarInitRequired());
                           },
                           &SynapseGroupInternal::updateWUInitHash);

    LOGD_CODE_GEN << "Merging synapse connectivity initialisation groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseConnectivityInitGroups,
                           [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
                           &SynapseGroupInternal::updateConnectivityInitHash);

    LOGD_CODE_GEN << "Merging synapse sparse initialization groups:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseSparseInitGroups,
                           [&backend](const SynapseGroupInternal &sg)
                           {
                               return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && 
                                       (sg.isWUVarInitRequired()
                                        || backend.isSynRemapRequired(sg)
                                        || (backend.isPostsynapticRemapRequired() && !sg.getWUModel()->getLearnPostCode().empty())));
                           },
                           &SynapseGroupInternal::updateWUInitHash);

    LOGD_CODE_GEN << "Merging custom sparse weight update initialization groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomWUUpdateSparseInitGroups,
                           [](const CustomUpdateWUInternal &cg) { return (cg.getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) && cg.isVarInitRequired(); },
                           &CustomUpdateWUInternal::updateInitHash);

    LOGD_CODE_GEN << "Merging neuron groups which require their spike queues updating:";
    createMergedGroups(model, backend, model.getNeuronGroups(), m_MergedNeuronSpikeQueueUpdateGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b)
                       {
                           return ((a.getNumDelaySlots() == b.getNumDelaySlots())
                                   && (a.isSpikeEventRequired() == b.isSpikeEventRequired())
                                   && (a.isTrueSpikeRequired() == b.isTrueSpikeRequired()));
                       });

    LOGD_CODE_GEN << "Merging neuron groups which require their previous spike times updating:";
    createMergedGroups(model, backend, model.getNeuronGroups(), m_MergedNeuronPrevSpikeTimeUpdateGroups,
                       [](const NeuronGroupInternal &ng){ return (ng.isPrevSpikeTimeRequired() || ng.isPrevSpikeEventTimeRequired()); },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b)
                       {
                           return ((a.getNumDelaySlots() == b.getNumDelaySlots())
                                   && (a.isSpikeEventRequired() == b.isSpikeEventRequired())
                                   && (a.isTrueSpikeRequired() == b.isTrueSpikeRequired())
                                   && (a.isPrevSpikeTimeRequired() == b.isPrevSpikeTimeRequired())
                                   && (a.isPrevSpikeEventTimeRequired() == b.isPrevSpikeEventTimeRequired()));
                       });

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
    createMergedGroups(model, backend, synapseGroupsWithDendriticDelay, m_MergedSynapseDendriticDelayUpdateGroups,
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b)
                       {
                           return (a.getMaxDendriticDelayTimesteps() == b.getMaxDendriticDelayTimesteps());
                       });

    LOGD_CODE_GEN << "Merging synapse groups which require host code to initialise their synaptic connectivity:";
    createMergedGroupsHash(model, backend, model.getSynapseGroups(), m_MergedSynapseConnectivityHostInitGroups,
                           [](const SynapseGroupInternal &sg)
                           { 
                               return (!sg.isWeightSharingSlave() && !sg.getConnectivityInitialiser().getSnippet()->getHostInitCode().empty()); 
                           },
                           &SynapseGroupInternal::updateConnectivityHostInitHash);

    LOGD_CODE_GEN << "Merging custom update groups:";
    createMergedGroupsHash(model, backend, model.getCustomUpdates(), m_MergedCustomUpdateGroups,
                           [](const CustomUpdateInternal &) { return true; },
                           &CustomUpdateInternal::updateHash);

    LOGD_CODE_GEN << "Merging custom weight update groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomUpdateWUGroups,
                           [](const CustomUpdateWUInternal &cg) { return !cg.isTransposeOperation(); },
                           &CustomUpdateWUInternal::updateHash);

    LOGD_CODE_GEN << "Merging custom weight transpose update groups:";
    createMergedGroupsHash(model, backend, model.getCustomWUUpdates(), m_MergedCustomUpdateTransposeWUGroups,
                           [](const CustomUpdateWUInternal &cg) { return cg.isTransposeOperation(); },
                           &CustomUpdateWUInternal::updateHash);

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