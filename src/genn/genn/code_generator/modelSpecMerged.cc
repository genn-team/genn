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
        mergedGroups.emplace_back(mergedGroups.size(), mergeTargets);

    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// CodeGenerator::ModelSpecMerged
//----------------------------------------------------------------------------
CodeGenerator::ModelSpecMerged::ModelSpecMerged(const ModelSpecInternal &model, const BackendBase &backend)
:   m_Model(model)
{
    LOGD << "Merging neuron update groups:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronUpdateGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b){ return a.canBeMerged(b); });

    LOGD << "Merging presynaptic update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedPresynapticUpdateGroups,
                       [](const SynapseGroupInternal &sg){ return (sg.isSpikeEventRequired() || sg.isTrueSpikeRequired()); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD << "Merging postsynaptic update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedPostsynapticUpdateGroups,
                       [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getLearnPostCode().empty(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD << "Merging synapse dynamics update groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseDynamicsGroups,
                       [](const SynapseGroupInternal &sg){ return !sg.getWUModel()->getSynapseDynamicsCode().empty(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canWUBeMerged(b); });

    LOGD << "Merging neuron initialization groups:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronInitGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b){ return a.canInitBeMerged(b); });

    LOGD << "Merging synapse dense initialization groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseDenseInitGroups,
                       [](const SynapseGroupInternal &sg)
                       {
                           return ((sg.getMatrixType() & SynapseMatrixConnectivity::DENSE) && sg.isWUVarInitRequired());
                       },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canInitBeMerged(b); });

    LOGD << "Merging synapse connectivity initialisation groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseConnectivityInitGroups,
                       [](const SynapseGroupInternal &sg){ return sg.isSparseConnectivityInitRequired(); },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canConnectivityInitBeMerged(b); });

    LOGD << "Merging synapse sparse initialization groups:";
    createMergedGroups(model.getSynapseGroups(), m_MergedSynapseSparseInitGroups,
                       [&backend](const SynapseGroupInternal &sg)
                       {
                           return ((sg.getMatrixType() & SynapseMatrixConnectivity::SPARSE) && 
                                   (sg.isWUVarInitRequired()
                                    || (backend.isSynRemapRequired() && !sg.getWUModel()->getSynapseDynamicsCode().empty())
                                    || (backend.isPostsynapticRemapRequired() && !sg.getWUModel()->getLearnPostCode().empty())));
                       },
                       [](const SynapseGroupInternal &a, const SynapseGroupInternal &b){ return a.canInitBeMerged(b); });

    LOGD << "Merging neuron groups which require their spike queues updating:";
    createMergedGroups(model.getNeuronGroups(), m_MergedNeuronSpikeQueueUpdateGroups,
                       [](const NeuronGroupInternal &){ return true; },
                       [](const NeuronGroupInternal &a, const NeuronGroupInternal &b)
                       {
                           return ((a.getNumDelaySlots() == b.getNumDelaySlots())
                                   && (a.isSpikeEventRequired() == b.isSpikeEventRequired())
                                   && (a.isTrueSpikeRequired() == b.isTrueSpikeRequired()));
                       });
}
