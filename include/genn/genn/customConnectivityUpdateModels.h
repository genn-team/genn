#pragma once

// GeNN includes
#include "gennExport.h"
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_PRE_VARS(...) virtual std::vector<NeuronVar> getPreVars() const override{ return __VA_ARGS__; }
#define SET_POST_VARS(...) virtual std::vector<NeuronVar> getPostVars() const override{ return __VA_ARGS__; }

#define SET_VAR_REFS(...) virtual VarRefVec getVarRefs() const override{ return __VA_ARGS__; }
#define SET_PRE_VAR_REFS(...) virtual VarRefVec getPreVarRefs() const override{ return __VA_ARGS__; }
#define SET_POST_VAR_REFS(...) virtual VarRefVec getPostVarRefs() const override{ return __VA_ARGS__; }

#define SET_ROW_UPDATE_CODE(ROW_UPDATE_CODE) virtual std::string getRowUpdateCode() const override{ return ROW_UPDATE_CODE; }
#define SET_HOST_UPDATE_CODE(HOST_UPDATE_CODE) virtual std::string getHostUpdateCode() const override{ return HOST_UPDATE_CODE; }

//----------------------------------------------------------------------------
// GeNN::CustomConnectivityUpdateModels::Base
//----------------------------------------------------------------------------
namespace GeNN::CustomConnectivityUpdateModels
{
//! Base class for all current source models
class GENN_EXPORT Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets model variables
    virtual std::vector<SynapseVar> getVars() const{ return {}; }

    //! Gets names and types (as strings) of synapse variable references
    virtual VarRefVec getVarRefs() const { return {}; }

    //! Gets names and types (as strings) of presynaptic variable references
    virtual VarRefVec getPreVarRefs() const { return {}; }

    //! Gets names and types (as strings) of postsynaptic variable references
    virtual VarRefVec getPostVarRefs() const { return {}; }

    //! Gets the code that performs a row-wise update 
    virtual std::string getRowUpdateCode() const { return ""; }

    //! Gets the code that performs host update 
    virtual std::string getHostUpdateCode() const { return ""; }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Find the index of a named presynaptic variable
    size_t getVarIndex(const std::string &varName) const
    {
        return getNamedVecIndex(varName, getVars());

    }

    //! Gets per-synapse model variables 
    /*! these have both VarAccessDim::PRE_NEURON and VarAccessDim::POST_NEURON */
    std::vector<SynapseVar> getSynVars() const;

    //! Gets presynaptic model variables 
    /*! these have VarAccessDim::PRE_NEURON and not VarAccessDim::POST_NEURON */
    std::vector<SynapseVar> getPreVars() const;

    //! Gets postsynaptic model variables 
    /*! these have VarAccessDim::POST_NEURON and not VarAccessDim::PRE_NEURON */
    std::vector<SynapseVar> getPostVars() const;

    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, double> &paramValues, 
                  const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                  const std::unordered_map<std::string, Models::VarReference> &varRefTargets,
                  const std::string &description) const;
};
}   // GeNN::CustomConnectivityUpdateModels
