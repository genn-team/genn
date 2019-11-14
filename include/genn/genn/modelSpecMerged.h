#pragma once

// Standard includes
#include <vector>

// GeNN includes
#include "neuronGroupMerged.h"
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
    const std::vector<NeuronGroupMerged> &getMergedLocalNeuronGroups() const { return m_MergedLocalNeuronGroups; }

    const ModelSpecInternal &getModel() const { return m_Model; }

    //! Get the string literal that should be used to represent a value in the model's floating-point type
    std::string scalarExpr(double val) const { return m_Model.scalarExpr(val); }

    std::string getPrecision() const { return m_Model.getPrecision(); }
    
    std::string getTimePrecision() const { return m_Model.getTimePrecision(); }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const ModelSpecInternal &m_Model;

    std::vector<NeuronGroupMerged> m_MergedLocalNeuronGroups;
};