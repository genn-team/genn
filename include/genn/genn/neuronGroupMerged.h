#pragma once

// Standard C++ includes
#include <functional>
#include <vector>

// GeNN includes
#include "neuronGroupInternal.h"

//----------------------------------------------------------------------------
// NeuronGroupMerged
//----------------------------------------------------------------------------
//! Very thin wrapper around a number of neuron groups which have been merged together
class NeuronGroupMerged
{
public:
    NeuronGroupMerged(size_t index, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &neuronGroups)
    :   m_Index(index), m_NeuronGroups(neuronGroups)
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getIndex() const { return m_Index; }

    //! Get 'archetype' neuron group - it's properties represent those of all other merged neuron groups
    const NeuronGroupInternal &getArchetype() const { return m_NeuronGroups.front().get(); }

    //! Gets access to underlying vector of neuron groups which have been merged
    const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &getNeuronGroups() const 
    {
        return m_NeuronGroups; 
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const size_t m_Index;
    std::vector<std::reference_wrapper<const NeuronGroupInternal>> m_NeuronGroups;
};
