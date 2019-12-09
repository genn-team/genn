#pragma once

// Standard includes
#include <functional>
#include <vector>

// GeNN includes
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

    GroupMerged(size_t index, const std::vector<std::reference_wrapper<const GroupInternal>> &groups)
    :   m_Index(index), m_Groups(groups)
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getIndex() const { return m_Index; }

    //! Get 'archetype' neuron group - it's properties represent those of all other merged neuron groups
    const GroupInternal &getArchetype() const { return m_Groups.front().get(); }

    //! Gets access to underlying vector of neuron groups which have been merged
    const std::vector<std::reference_wrapper<const GroupInternal>> &getGroups() const{ return m_Groups; }

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
class NeuronGroupMerged : public GroupMerged<NeuronGroupInternal>
{
public:
    NeuronGroupMerged(size_t index, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups)
    :   GroupMerged<NeuronGroupInternal>(index, groups)
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Find the synapse group compatible with specified merged insyn in archetype
    /*! **NOTE** this should only be called with neuron groups within merged group */
    const SynapseGroupInternal *getCompatibleMergedInSyn(size_t archetypeMergedInSyn, const NeuronGroupInternal &ng) const;

    //! Find the synapse group compatible with specified post in archetype
    /*! **NOTE** this should only be called with neuron groups within merged group */
    const SynapseGroupInternal *getCompatibleInitMergedInSyn(size_t archetypeMergedInSyn, const NeuronGroupInternal &ng) const;

    //! Find the synapse group compatible with specified merged insyn with post code in archetype
    /*! **NOTE** this should only be called with neuron groups within merged group */
    const SynapseGroupInternal *getCompatibleInSynWithPostCode(size_t archetypeInSynWithPostCode, const NeuronGroupInternal &ng) const;

    //! Find the synapse group compatible with specified insyn with post code in archetype
    /*! **NOTE** this should only be called with neuron groups within merged group */
    const SynapseGroupInternal *getCompatibleInitInSynWithPostCode(size_t archetypeInSynWithPostCode, const NeuronGroupInternal &ng) const;

    //! Find the current source compatible with specified current source in archetype
    /*! **NOTE** this should only be called with neuron groups within merged group */
    const CurrentSourceInternal *getCompatibleCurrentSource(size_t archetypeCurrentSource, const NeuronGroupInternal &ng) const;

    //! Find the current source compatible with specified current source in archetype
    /*! **NOTE** this should only be called with neuron groups within merged group */
    const CurrentSourceInternal *getCompatibleInitCurrentSource(size_t archetypeCurrentSource, const NeuronGroupInternal &ng) const;
    
    //! Get the expression to calculate the queue offset for accessing state of variables this timestep
    std::string getCurrentQueueOffset() const;

    //! Get the expression to calculate the queue offset for accessing state of variables in previous timestep
    std::string getPrevQueueOffset() const;

};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMerged
//----------------------------------------------------------------------------
class SynapseGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseGroupMerged(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
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

};
}   // namespace CodeGenerator
