#pragma once

// Standard includes
#include <vector>

// GeNN includes
#include "neuronGroupMerged.h"
#include "synapseGroupMerged.h"
#include "modelSpecInternal.h"

// Forward declarations
class ModelSpecInternal;

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

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const ModelSpecInternal &m_Model;

    std::vector<NeuronGroupMerged> m_MergedLocalNeuronGroups;
    std::vector<SynapseGroupMerged> m_MergedLocalSynapseGroups;
};
