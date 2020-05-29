#include "code_generator/modelSpecMerged.h"

// GeNN includes
#include "logging.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
template<typename Group, typename MergedGroup, typename M>
void createMergedGroups(std::vector<std::reference_wrapper<const Group>> &unmergedGroups, 
                        std::vector<MergedGroup> &mergedGroups, M canMerge)
{
    // Loop through un-merged  groups
    std::vector<std::vector<std::reference_wrapper<const Group>>> protoMergedGroups;
    while(!unmergedGroups.empty()) {
        // Remove last group from vector
        const Group &group = unmergedGroups.back().get();
        unmergedGroups.pop_back();

        // Loop through existing proto-merged groups
        bool existingMergedGroupFound = false;
        for(auto &p : protoMergedGroups) {
            assert(!p.empty());

            // If our group can be merged with this proto-merged group
            if(canMerge(p.front().get(), group)) {
                // Add group to vector
                p.emplace_back(group);

                // Set flag and stop searching
                existingMergedGroupFound = true;
                break;
            }
        }

        // If no existing merged groups were found, 
        // create a new proto-merged group containing just this group
        if(!existingMergedGroupFound) {
            protoMergedGroups.emplace_back();
            protoMergedGroups.back().emplace_back(group);
        }
    }

    // Reserve final merged groups vector
    mergedGroups.reserve(protoMergedGroups.size());

    // Build, moving vectors of groups into data structure to avoid copying
    for(size_t i = 0; i < protoMergedGroups.size(); i++) {
        mergedGroups.emplace_back(i, std::move(protoMergedGroups[i]));
    }
}
//----------------------------------------------------------------------------
template<typename Group, typename MergedGroup, typename F, typename M>
void createMergedGroups(const std::map<std::string, Group> &groups, std::vector<MergedGroup> &mergedGroups,
                        F filter, M canMerge)
{
    // Build temporary vector of references to groups that pass filter
    std::vector<std::reference_wrapper<const Group>> unmergedGroups;
    for(const auto &g : groups) {
        if(filter(g.second)) {
            unmergedGroups.emplace_back(std::cref(g.second));
        }
    }

    // Merge filtered vector
    createMergedGroups(unmergedGroups, mergedGroups, canMerge);
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// CodeGenerator::ModelSpecMerged
//----------------------------------------------------------------------------
CodeGenerator::ModelSpecMerged::ModelSpecMerged(const ModelSpecInternal &model, const BackendBase &backend)
:   m_Model(model), m_NeuronUpdateSupportCode("NeuronUpdateSupportCode"), m_PostsynapticDynamicsSupportCode("PostsynapticDynamicsSupportCode"),
    m_PresynapticUpdateSupportCode("PresynapticUpdateSupportCode"), m_PostsynapticUpdateSupportCode("PostsynapticUpdateSupportCode"),
    m_SynapseDynamicsSupportCode("SynapseDynamicsSupportCode")
{
    LOGD_CODE_GEN << "Merging neuron update groups:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronUpdateGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b){ return a.canBeMerged(b); });

    LOGD_CODE_GEN << "Merging presynaptic update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedPresynapticUpdateGroups,
                       [](const SynapseGroupInternal &sg){ return (sg.isSpikeEventRequired() || sg.isTrueSpikeRequired()); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD_CODE_GEN << "Merging postsynaptic update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedPostsynapticUpdateGroups,
                       [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getLearnPostCode().empty(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD_CODE_GEN << "Merging synapse dynamics update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseDynamicsGroups,
                       [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD_CODE_GEN << "Merging neuron initialization groups:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronInitGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b){ return a.canInitBeMerged(b); });

    LOGD_CODE_GEN << "Merging synapse dense initialization groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseDenseInitGroups,
                       [](const SynapseGroupInternal &sg)
                       {
                           return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && sg.isWUVarInitRequired());
                       },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUInitBeMerged(b); });

    LOGD_CODE_GEN << "Merging synapse connectivity initialisation groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseConnectivityInitGroups,
                       [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canConnectivityInitBeMerged(b); });

    LOGD_CODE_GEN << "Merging synapse sparse initialization groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseSparseInitGroups,
                       [&backend](const SynapseGroupInternal &sg)
                       {
                           return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && 
                                   (sg.isWUVarInitRequired()
                                    || (backend.isSynRemapRequired() && !sg.getWUModel()->getSynapseDynamicsCode().empty())
                                    || (backend.isPostsynapticRemapRequired() && !sg.getWUModel()->getLearnPostCode().empty())));
                       },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUInitBeMerged(b); });

    LOGD_CODE_GEN << "Merging neuron groups which require their spike queues updating:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronSpikeQueueUpdateGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b)
                       {
                           return ((a.getNumDelaySlots() == b.getNumDelaySlots())
                                   && (a.isSpikeEventRequired() == b.isSpikeEventRequired())
                                   && (a.isTrueSpikeRequired() == b.isTrueSpikeRequired()));
                       });

    // Build vector of merged synapse groups which require dendritic delay
    std::vector<std::reference_wrapper<const SynapseGroupInternal>> synapseGroupsWithDendriticDelay;
    for(const auto &n : model.getNeuronGroups()) {
        for(const auto &m : n.second.getMergedInSyn()) {
            if(m.first->isDendriticDelayRequired()) {
                synapseGroupsWithDendriticDelay.push_back(std::cref(*m.first));
            }
        }
    }
    LOGD_CODE_GEN << "Merging synapse groups which require their dendritic delay updating:";
    createMergedGroups(synapseGroupsWithDendriticDelay, m_MergedSynapseDendriticDelayUpdateGroups,
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b)
                       {
                           return (a.getMaxDendriticDelayTimesteps() == b.getMaxDendriticDelayTimesteps());
                       });

    LOGD_CODE_GEN << "Merging synapse groups which require host code to initialise their synaptic connectivity:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseConnectivityHostInitGroups,
                       [](const SynapseGroupInternal &sg)
                       { 
                           return (!sg.isWeightSharingSlave() && !sg.getConnectivityInitialiser().getSnippet()->getHostInitCode().empty()); 
                       },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b)
                       { 
                           return a.canConnectivityHostInitBeMerged(b); 
                       });

    // Loop through merged neuron groups
    for(const auto &ng : m_MergedNeuronUpdateGroups) {
        // Add neuron support code
        m_NeuronUpdateSupportCode.addSupportCode(ng.getArchetype().getNeuronModel()->getSupportCode());

        // Loop through merged postsynaptic models and add their support code
        for(const auto &sg : ng.getArchetype().getMergedInSyn()) {
            m_PostsynapticDynamicsSupportCode.addSupportCode(sg.first->getPSModel()->getSupportCode());
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
