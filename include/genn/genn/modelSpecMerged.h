#pragma once

// Standard includes
#include <vector>

// GeNN includes
#include "modelSpecInternal.h"

// Forward declarations
class ModelSpecInternal;

//----------------------------------------------------------------------------
// GroupMerged
//----------------------------------------------------------------------------
//! Very thin wrapper around a number of groups which have been merged together
template<typename GroupInternal>
class GroupMerged
{
public:
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

typedef GroupMerged<NeuronGroupInternal> NeuronGroupMerged;
typedef GroupMerged<SynapseGroupInternal> SynapseGroupMerged;

//----------------------------------------------------------------------------
// ModelSpecMerged
//----------------------------------------------------------------------------
class ModelSpecMerged
{
public:
    ModelSpecMerged(const ModelSpecInternal &model);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const std::vector<NeuronGroupMerged> &getMergedLocalNeuronGroups() const{ return m_MergedLocalNeuronGroups; }
    const std::vector<SynapseGroupMerged> &getMergedLocalSynapseGroups() const{ return m_MergedLocalSynapseGroups; }

    const ModelSpecInternal &getModel() const{ return m_Model; }

    //! Get the string literal that should be used to represent a value in the model's floating-point type
    std::string scalarExpr(double val) const{ return m_Model.scalarExpr(val); }

    std::string getPrecision() const{ return m_Model.getPrecision(); }
    
    std::string getTimePrecision() const{ return m_Model.getTimePrecision(); }

    double getDT() const{ return m_Model.getDT(); }

    bool isTimingEnabled() const{ return m_Model.isTimingEnabled(); }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const ModelSpecInternal &m_Model;

    std::vector<NeuronGroupMerged> m_MergedLocalNeuronGroups;
    std::vector<SynapseGroupMerged> m_MergedLocalSynapseGroups;
};
