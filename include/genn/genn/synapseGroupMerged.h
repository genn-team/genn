#pragma once

// Standard C++ includes
#include <functional>
#include <vector>

// GeNN includes
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// SynapseGroupMerged
//----------------------------------------------------------------------------
//! Very thin wrapper around a number of synapse groups which have been merged together
class SynapseGroupMerged
{
public:
    SynapseGroupMerged(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &synapseGroups)
    :   m_Index(index), m_SynapseGroups(synapseGroups)
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getIndex() const { return m_Index; }

    //! Get 'archetype' neuron group - it's properties represent those of all other merged neuron groups
    const SynapseGroupInternal &getArchetype() const { return m_SynapseGroups.front().get(); }

    //! Gets access to underlying vector of synapse groups which have been merged
    const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &getSynapseGroups() const{ return m_SynapseGroups; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const size_t m_Index;
    std::vector<std::reference_wrapper<const SynapseGroupInternal>> m_SynapseGroups;
};
