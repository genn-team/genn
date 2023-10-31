#pragma once

// GeNN includes
#include "gennExport.h"
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_CUSTOM_UPDATE_VARS(...) virtual std::vector<CustomUpdateVar> getVars() const override{ return __VA_ARGS__; }
#define SET_VAR_REFS(...) virtual VarRefVec getVarRefs() const override{ return __VA_ARGS__; }
#define SET_EXTRA_GLOBAL_PARAM_REFS(...) virtual EGPRefVec getExtraGlobalParamRefs() const override{ return __VA_ARGS__; }
#define SET_UPDATE_CODE(UPDATE_CODE) virtual std::string getUpdateCode() const override{ return UPDATE_CODE; }


//----------------------------------------------------------------------------
// GeNN::CustomUpdateModels::Base
//----------------------------------------------------------------------------
namespace GeNN::CustomUpdateModels
{
//! Base class for all current source models
class GENN_EXPORT Base : public Models::Base
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    //! Gets model variables
    virtual std::vector<CustomUpdateVar> getVars() const{ return {}; }

    //! Gets names and typesn of model variable references
    virtual VarRefVec getVarRefs() const{ return {}; }

    //! Gets names and types of model extra global parameter references
    virtual EGPRefVec getExtraGlobalParamRefs() const { return {}; }

    //! Gets the code that performs the custom update 
    virtual std::string getUpdateCode() const{ return ""; }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Find the index of a named variable
    size_t getVarIndex(const std::string &varName) const
    {
        return getNamedVecIndex(varName, getVars());
    }

    //! Validate names of parameters etc
    void validate(const std::unordered_map<std::string, double> &paramValues,
                  const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                  const std::unordered_map<std::string, Models::VarReference> &varRefTargets,
                  const std::string &description) const;

    void validate(const std::unordered_map<std::string, double> &paramValues,
                  const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                  const std::unordered_map<std::string, Models::WUVarReference> &varRefTargets,
                  const std::string &description) const;
};

//----------------------------------------------------------------------------
// CustomUpdateModels::Transpose
//----------------------------------------------------------------------------
//! Minimal custom update model for calculating tranpose
class Transpose : public Base
{
    DECLARE_SNIPPET(Transpose);

    SET_VAR_REFS({{"variable", "scalar", VarAccessMode::READ_ONLY}});
};
}   // GeNN::CustomUpdateModels

