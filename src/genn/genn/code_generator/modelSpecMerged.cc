#include "code_generator/modelSpecMerged.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
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
                        std::vector<MergedGroup> &mergedGroups, bool init, M canMerge)
{
    // Loop through un-merged  groups
    while(!unmergedGroups.empty()) {
        // Remove last group from vector
        const Group &group = unmergedGroups.back().get();
        unmergedGroups.pop_back();

        // Start vector of groups that can be merged
        std::vector<std::reference_wrapper<const Group>> mergeTargets{group};

        // Loop through other remaining unmerged groups
        for(auto otherGroup = unmergedGroups.begin(); otherGroup != unmergedGroups.end();) {
            // If this 'other' group can be merged with original
            if(canMerge(group, otherGroup->get())) {
                LOGD << "\tMerging group '" << otherGroup->get().getName() << "' with '" << group.getName() << "'";

                // Add to list of merge targets
                mergeTargets.push_back(otherGroup->get());

                // Remove from unmerged list
                otherGroup = unmergedGroups.erase(otherGroup);
            }
            // Otherwise, advance to next group
            else {
                LOGD << "\tUnable to merge group '" << otherGroup->get().getName() << "' with '" << group.getName() << "'";
                ++otherGroup;
            }
        }

        // A new merged neuron group to model
        mergedGroups.emplace_back(mergedGroups.size(), init, mergeTargets);

    }
}
//----------------------------------------------------------------------------
template<typename Group, typename MergedGroup, typename F, typename M>
void createMergedGroups(const std::map<std::string, Group> &groups, std::vector<MergedGroup> &mergedGroups, bool init,
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
    createMergedGroups(unmergedGroups, mergedGroups, init, canMerge);
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
    LOGD << "Merging neuron update groups:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronUpdateGroups, false,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b){ return a.canBeMerged(b); });

    LOGD << "Merging presynaptic update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedPresynapticUpdateGroups, false,
                       [](const SynapseGroupInternal &sg){ return (sg.isSpikeEventRequired() || sg.isTrueSpikeRequired()); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD << "Merging postsynaptic update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedPostsynapticUpdateGroups, false,
                       [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getLearnPostCode().empty(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD << "Merging synapse dynamics update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseDynamicsGroups, false,
                       [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD << "Merging neuron initialization groups:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronInitGroups, true,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b){ return a.canInitBeMerged(b); });

    LOGD << "Merging synapse dense initialization groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseDenseInitGroups, true,
                       [](const SynapseGroupInternal &sg)
                       {
                           return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && sg.isWUVarInitRequired());
                       },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUInitBeMerged(b); });

    LOGD << "Merging synapse connectivity initialisation groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseConnectivityInitGroups, true,
                       [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canConnectivityInitBeMerged(b); });

    LOGD << "Merging synapse sparse initialization groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseSparseInitGroups, true,
                       [&backend](const SynapseGroupInternal &sg)
                       {
                           return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && 
                                   (sg.isWUVarInitRequired()
                                    || (backend.isSynRemapRequired() && !sg.getWUModel()->getSynapseDynamicsCode().empty())
                                    || (backend.isPostsynapticRemapRequired() && !sg.getWUModel()->getLearnPostCode().empty())));
                       },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUInitBeMerged(b); });

    LOGD << "Merging neuron groups which require their spike queues updating:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronSpikeQueueUpdateGroups, false,
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
    LOGD << "Merging synapse groups which require their dendritic delay updating:";
    createMergedGroups(synapseGroupsWithDendriticDelay, m_MergedSynapseDendriticDelayUpdateGroups, false,
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b)
                       {
                           return (a.getMaxDendriticDelayTimesteps() == b.getMaxDendriticDelayTimesteps());
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
