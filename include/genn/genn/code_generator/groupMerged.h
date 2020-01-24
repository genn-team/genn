#pragma once

// Standard includes
#include <functional>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// CodeGenerator::GroupMerged
//----------------------------------------------------------------------------
//! Very thin wrapper around a number of groups which have been merged together
namespace CodeGenerator
{
template<typename G>
class GroupMerged
{
public:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef G GroupInternal;

    GroupMerged(size_t index, const std::vector<std::reference_wrapper<const GroupInternal>> groups)
    :   m_Index(index), m_Groups(std::move(groups))
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getIndex() const { return m_Index; }

    //! Get 'archetype' neuron group - it's properties represent those of all other merged neuron groups
    const GroupInternal &getArchetype() const { return m_Groups.front().get(); }

    //! Gets access to underlying vector of neuron groups which have been merged
    const std::vector<std::reference_wrapper<const GroupInternal>> &getGroups() const{ return m_Groups; }

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    //! Helper to test whether parameter values are heterogeneous within merged group
    template<typename P>
    bool isParamValueHeterogeneous(size_t index, P getParamValuesFn) const
    {
        // Get value of parameter in archetype group
        const double archetypeValue = getParamValuesFn(getArchetype()).at(index);

        // Return true if any parameter values differ from the archetype value
        return std::any_of(getGroups().cbegin(), getGroups().cend(),
                           [archetypeValue, index, getParamValuesFn](const GroupInternal &g)
                           {
                               return (getParamValuesFn(g).at(index) != archetypeValue);
                           });
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const size_t m_Index;
    std::vector<std::reference_wrapper<const GroupInternal>> m_Groups;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronGroupMerged : public GroupMerged<NeuronGroupInternal>
{
public:
    NeuronGroupMerged(size_t index, bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get the expression to calculate the queue offset for accessing state of variables this timestep
    std::string getCurrentQueueOffset() const;

    //! Get the expression to calculate the queue offset for accessing state of variables in previous timestep
    std::string getPrevQueueOffset() const;

    //! Is the parameter implemented as a heterogeneous parameter?
    bool isParamHeterogeneous(size_t index) const;

    //! Is the derived parameter implemented as a heterogeneous parameter?
    bool isDerivedParamHeterogeneous(size_t index) const;

    //! Is the current source parameter implemented as a heterogeneous parameter?
    bool isCurrentSourceParamHeterogeneous(size_t childIndex, size_t paramIndex) const
    {
        return isChildParamHeterogeneous(childIndex, paramIndex, m_SortedCurrentSources,
                                       [](const CurrentSourceInternal *cs) { return cs->getParams();  });
    }

    //! Is the current source derived parameter implemented as a heterogeneous parameter?
    bool isCurrentSourceDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const
    {
        return isChildParamHeterogeneous(childIndex, paramIndex, m_SortedCurrentSources,
                                       [](const CurrentSourceInternal *cs) { return cs->getDerivedParams();  });
    }

    const std::vector<std::vector<std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>>>> &getSortedMergedInSyns() const{ return m_SortedMergedInSyns; }
    const std::vector<std::vector<CurrentSourceInternal *>> &getSortedCurrentSources() const { return m_SortedCurrentSources; }
    const std::vector<std::vector<SynapseGroupInternal *>> &getSortedInSynWithPostCode() const { return m_SortedInSynWithPostCode; }
    const std::vector<std::vector<SynapseGroupInternal *>> &getSortedOutSynWithPreCode() const{ return m_SortedOutSynWithPreCode; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    template<typename T, typename G, typename C>
    void orderNeuronGroupChildren(const std::vector<T> &archetypeChildren,
                                  std::vector<std::vector<T>> &sortedGroupChildren,
                                  G getVectorFunc, C isCompatibleFunc) const
    {
        // Reserve vector of vectors to hold children for all neuron groups, in archetype order
        sortedGroupChildren.reserve(archetypeChildren.size());

        // Loop through groups
        for(const auto &g : getGroups()) {
            // Make temporary copy of this group's children
            std::vector<T> tempChildren((g.get().*getVectorFunc)());

            assert(tempChildren.size() == archetypeChildren.size());

            // Reserve vector for this group's children
            sortedGroupChildren.emplace_back();
            sortedGroupChildren.back().reserve(tempChildren.size());

            // Loop through archetype group's children
            for(const auto &archetypeG : archetypeChildren) {
                // Find compatible child in temporary list
                const auto otherChild = std::find_if(tempChildren.cbegin(), tempChildren.cend(),
                                                     [archetypeG, isCompatibleFunc](const T &g)
                                                     {
                                                         return isCompatibleFunc(archetypeG, g);
                                                     });
                assert(otherChild != tempChildren.cend());

                // Add pointer to vector of compatible merged in syns
                sortedGroupChildren.back().push_back(*otherChild);

                // Remove from original vector
                tempChildren.erase(otherChild);
            }
        }
    }
    
    template<typename T, typename G, typename C>
    void orderNeuronGroupChildren(std::vector<std::vector<T>> &sortedGroupChildren,
                                  G getVectorFunc, C isCompatibleFunc) const
    {
        const std::vector<T> &archetypeChildren = (getArchetype().*getVectorFunc)();
        orderNeuronGroupChildren(archetypeChildren, sortedGroupChildren, getVectorFunc, isCompatibleFunc);
    }
    
    template<typename T, typename G>
    bool isChildParamHeterogeneous(size_t childIndex, size_t paramIndex, const std::vector<std::vector<T>> &sortedGroupChildren,
                                   G getParamValuesFn) const
    {
        // Get value of archetype derived parameter
        const double firstValue = getParamValuesFn(sortedGroupChildren[0][childIndex]).at(paramIndex);

        // Loop through groups within merged group
        for(size_t i = 0; i < sortedGroupChildren.size(); i++) {
            const auto group = sortedGroupChildren[i][childIndex];
            if(getParamValuesFn(group).at(paramIndex) != firstValue) {
                return true;
            }
        }
        return false;
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>>>> m_SortedMergedInSyns;
    std::vector<std::vector<CurrentSourceInternal*>> m_SortedCurrentSources;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedInSynWithPostCode;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedOutSynWithPreCode;
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseGroupMerged(size_t index, bool, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   GroupMerged<SynapseGroupInternal>(index, groups)
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get the expression to calculate the delay slot for accessing
    //! Presynaptic neuron state variables, taking into account axonal delay
    std::string getPresynapticAxonalDelaySlot() const;

    //! Get the expression to calculate the delay slot for accessing
    //! Postsynaptic neuron state variables, taking into account back propagation delay
    std::string getPostsynapticBackPropDelaySlot() const;

    std::string getDendriticDelayOffset(const std::string &offset = "") const;

    //! Is the weight update model variable initialization parameter implemented as a heterogeneous parameter?
    bool isWUVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const;
    
    //! Is the weight update model variable initialization derived parameter implemented as a heterogeneous parameter?
    bool isWUVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const;
};
}   // namespace CodeGenerator
