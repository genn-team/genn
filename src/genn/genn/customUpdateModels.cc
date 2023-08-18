#include "customUpdateModels.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::CustomUpdateModels
{
// Implement models
IMPLEMENT_SNIPPET(Transpose);

//----------------------------------------------------------------------------
// GeNN::CustomUpdateModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getUpdateCode(), hash);
    Utils::updateHash(getVarRefs(), hash);
    Utils::updateHash(getExtraGlobalParamRefs(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues,
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::unordered_map<std::string, Models::VarReference> &varRefTargets,
                    const std::string &description) const
{
     // Superclass
    Models::Base::validate(paramValues, varValues, description);

    const auto varRefs = getVarRefs();
    Utils::validateVecNames(varRefs, "Variable reference");

    // Validate variable reference initialisers
    Utils::validateInitialisers(varRefs, varRefTargets, "Variable reference", description);
    Utils::validateVecNames(getExtraGlobalParamRefs(), "Extra global parameter reference");
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues,
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::unordered_map<std::string, Models::WUVarReference> &varRefTargets,
                    const std::string &description) const
{
     // Superclass
    Models::Base::validate(paramValues, varValues, description);

    const auto varRefs = getVarRefs();
    Utils::validateVecNames(getVarRefs(), "Variable reference");

    // Validate variable reference initialisers
    Utils::validateInitialisers(varRefs, varRefTargets, "Variable reference", description);
    Utils::validateVecNames(getExtraGlobalParamRefs(), "Extra global parameter reference");

    // If any variables have an invalid access mode, give an error
    const auto vars = getVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                   [](const Models::Base::Var &v){ return !v.access.template isValid<CustomUpdateVarAccess>(); }))
    {
        throw std::runtime_error("Custom update model variables must have CustomUpdateVarAccess access type");
    }
}
}   // namespace GeNN::CustomUpdateModels