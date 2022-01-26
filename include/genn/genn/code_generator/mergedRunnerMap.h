#pragma once

// Standard C++ includes
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// GeNN code generator includes
#include "runnerGroupMerged.h"


//--------------------------------------------------------------------------
// CodeGenerator::MergedRunnerMap
//--------------------------------------------------------------------------
namespace CodeGenerator
{
class MergedRunnerMap
{
public:
    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    template<typename MergedGroup>
    void addGroup(const std::vector<MergedGroup> &mergedGroups)
    {
        // Loop through merged groups
        for(const auto &g : mergedGroups) {
            // Loop through individual groups inside and add to map
            for(size_t i = 0; i < g.getGroups().size(); i++) {
                auto result = m_MergedRunnerGroups.emplace(std::piecewise_construct,
                                                           std::forward_as_tuple(g.getGroups().at(i).get().getName()),
                                                           std::forward_as_tuple(g.getIndex(), i));
                assert(result.second);
            }
        }
    }

    //! Find the name of the merged runner group associated with neuron group e.g. mergedNeuronRunnerGroup0[6]
    std::string findGroup(const NeuronGroupInternal &ng) const { return findGroup<NeuronRunnerGroupMerged>(ng); }

    //! Find the name of the merged runner group associated with synapse group e.g. mergedSynapseRunnerGroup0[6]
    std::string findGroup(const SynapseGroupInternal &sg) const { return findGroup<SynapseRunnerGroupMerged>(sg); }

    //! Find the name of the merged runner group associated with current source e.g. mergedCurrentSourceRunnerGroup0[6]
    std::string findGroup(const CurrentSourceInternal &cs) const { return findGroup<CurrentSourceRunnerGroupMerged>(cs); }

    //! Find the name of the merged runner group associated with custom update e.g. mergedCustomUpdateRunnerGroup0[6]
    std::string findGroup(const CustomUpdateInternal &cu) const { return findGroup<CustomUpdateRunnerGroupMerged>(cu); }

    //! Find the name of the merged runner group associated with custom update e.g. mergedCustomUpdateWURunnerGroup0[6]
    std::string findGroup(const CustomUpdateWUInternal &cu) const { return findGroup<CustomUpdateWURunnerGroupMerged>(cu); }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    //! Helper to find merged runner group
    template<typename MergedGroup>
    std::string findGroup(const typename MergedGroup::GroupInternal &g) const
    {
        // Find group by name
        const auto m = m_MergedRunnerGroups.at(g.getName());

        // Return structure
        return "merged" + MergedGroup::name + "Group" + std::to_string(std::get<0>(m)) + "[" + std::to_string(std::get<1>(m)) + "]";
    }


    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    //! Map of group names to index of merged group and index of group within that
    std::unordered_map<std::string, std::tuple<unsigned int, unsigned int>> m_MergedRunnerGroups;
};
}