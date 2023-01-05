#pragma once

// GeNN includes
#include "gennExport.h"
#include "models.h"

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define SET_VAR_REFS(...) virtual VarRefVec getVarRefs() const override{ return __VA_ARGS__; }
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
    //! Gets names and types (as strings) of model variable references
    virtual VarRefVec getVarRefs() const{ return {}; }

    //! Gets the code that performs the custom update 
    virtual std::string getUpdateCode() const{ return ""; }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Update hash from model
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //! Validate names of parameters etc
    template<typename R>
    void validate(const std::unordered_map<std::string, double> &paramValues,
                  const std::unordered_map<std::string, Models::VarInit> &varValues,
                  const std::unordered_map<std::string, R> &varRefTargets,
                  const std::string &description) const
    {
        // Superclass
        Models::Base::validate(paramValues, varValues, description);

        const auto varRefs = getVarRefs();
        Utils::validateVecNames(getVarRefs(), "Variable reference");

        // Validate variable reference initialisers
        Utils::validateInitialisers(varRefs, varRefTargets, "Variable reference", description);
    }
};

//----------------------------------------------------------------------------
// CustomUpdateModels::Transpose
//----------------------------------------------------------------------------
//! Minimal custom update model for calculating tranpose
class Transpose : public Base
{
    DECLARE_SNIPPET(Transpose);

    SET_VAR_REFS({{"variable", "scalar", VarAccessMode::READ_WRITE}});
};
}   // GeNN::CustomUpdateModels

