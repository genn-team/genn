#pragma once

// GeNN includes
#include "gennExport.h"
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_PRE_VARS(...) virtual std::vector<Var> getPreVars() const override{ return __VA_ARGS__; }
#define SET_POST_VARS(...) virtual std::vector<Var> getPostVars() const override{ return __VA_ARGS__; }

#define SET_VAR_REFS(...) virtual VarRefVec getVarRefs() const override{ return __VA_ARGS__; }
#define SET_PRE_VAR_REFS(...) virtual VarRefVec getPreVarRefs() const override{ return __VA_ARGS__; }
#define SET_POST_VAR_REFS(...) virtual VarRefVec getPostVarRefs() const override{ return __VA_ARGS__; }

#define SET_EXTRA_GLOBAL_PARAM_REFS(...) virtual EGPRefVec getExtraGlobalParamRefs() const override{ return __VA_ARGS__; }

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
    //! Gets names and types (as strings) of state variables that are common
    //! across all synapses coming from the same presynaptic neuron
    virtual std::vector<Var> getPreVars() const { return {}; }

    //! Gets names and types (as strings) of state variables that are common
    //! across all synapses going to the same postsynaptic neuron
    virtual std::vector<Var> getPostVars() const { return {}; }

    //! Gets model variables
    virtual std::vector<Var> getVars() const{ return {}; }

    //! Gets names and types (as strings) of synapse variable references
    virtual VarRefVec getVarRefs() const { return {}; }

    //! Gets names and types (as strings) of presynaptic variable references
    virtual VarRefVec getPreVarRefs() const { return {}; }

    //! Gets names and types (as strings) of postsynaptic variable references
    virtual VarRefVec getPostVarRefs() const { return {}; }

    //! Gets names and types of model extra global parameter references
    virtual EGPRefVec getExtraGlobalParamRefs() const { return {}; }

    //! Gets the code that performs a row-wise update 
    virtual std::string getRowUpdateCode() const { return ""; }

    //! Gets the code that performs host update 
    virtual std::string getHostUpdateCode() const { return ""; }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Find the named variable
    std::optional<Var> getVar(const std::string &varName) const
    {
        return getNamed(varName, getVars());
    }

    //! Find the named presynaptic variable
    std::optional<Var> getPreVar(const std::string &varName) const
    {
        return getNamed(varName, getPreVars());
    }

    //! Find the named postsynaptic variable
    std::optional<Var> getPostVar(const std::string &varName) const
    {
        return getNamed(varName, getPostVars());
    }

    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues, 
                  const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                  const std::unordered_map<std::string, InitVarSnippet::Init> &preVarValues,
                  const std::unordered_map<std::string, InitVarSnippet::Init> &postVarValues,
                  const std::unordered_map<std::string, Models::WUVarReference> &varRefTargets,
                  const std::unordered_map<std::string, Models::VarReference> &preVarRefTargets,
                  const std::unordered_map<std::string, Models::VarReference> &postVarRefTargets,
                  const std::unordered_map<std::string, Models::EGPReference> &egpRefTargets,
                  const std::string &description) const;
};
}   // GeNN::CustomConnectivityUpdateModels
