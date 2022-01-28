#pragma once

// Standard C++ includes
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Forward declarations
class NeuronGroup;
class SynapseGroup;
class CurrentSource;
class CustomUpdate;
class CustomUpdateWU;

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
    std::string getStruct(const NeuronGroup &ng) const;

    //! Find the name of the merged runner group associated with synapse group e.g. mergedSynapseRunnerGroup0[6]
    std::string getStruct(const SynapseGroup &sg) const;

    //! Find the name of the merged runner group associated with current source e.g. mergedCurrentSourceRunnerGroup0[6]
    std::string getStruct(const CurrentSource &cs) const;

    //! Find the name of the merged runner group associated with custom update e.g. mergedCustomUpdateRunnerGroup0[6]
    std::string getStruct(const CustomUpdate &cu) const;

    //! Find the name of the merged runner group associated with custom update e.g. mergedCustomUpdateWURunnerGroup0[6]
    std::string getStruct(const CustomUpdateWU &cu) const;

    std::tuple<size_t, size_t> getIndices(const std::string &name) const{ return m_MergedRunnerGroups.at(name); }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    //! Helper to find merged runner group
    template<typename MergedGroup>
    std::string getStruct(const std::string &name) const
    {
        // Find group by name
        const auto m = m_MergedRunnerGroups.at(name);

        // Return structure
        return "merged" + MergedGroup::name + "Group" + std::to_string(std::get<0>(m)) + "[" + std::to_string(std::get<1>(m)) + "]";
    }


    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    //! Map of group names to index of merged group and index of group within that
    std::unordered_map<std::string, std::tuple<size_t, size_t>> m_MergedRunnerGroups;
};
}
