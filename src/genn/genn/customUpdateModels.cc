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
    Snippet::Base::updateHash(hash);
    Utils::updateHash(getVars(), hash);
    Utils::updateHash(getUpdateCode(), hash);
    Utils::updateHash(getVarRefs(), hash);
    Utils::updateHash(getExtraGlobalParamRefs(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::map<std::string, Type::NumericValue> &paramValues,
                    const std::map<std::string, InitVarSnippet::Init> &varValues,
                    const std::map<std::string, Models::VarReference> &varRefTargets,
                    const std::map<std::string, Models::EGPReference> &egpRefTarget,
                    const std::string &description) const
{
     // Superclass
    Snippet::Base::validate(paramValues, description);

    // Validate variables
    const auto vars = getVars();
    Utils::validateVecNames(vars, "Variable");
    Utils::validateInitialisers(vars, varValues, "variable", description);

    // Validate variable references
    const auto varRefs = getVarRefs();
    Utils::validateVecNames(varRefs, "Variable reference");
    Utils::validateInitialisers(varRefs, varRefTargets, "Variable reference", description);

    // Validate EGP references
    const auto egpRefs = getExtraGlobalParamRefs();
    Utils::validateVecNames(egpRefs, "Extra global parameter reference");
    Utils::validateInitialisers(egpRefs, egpRefTarget, "Extra Global Parameter reference", description);
}
//----------------------------------------------------------------------------
void Base::validate(const std::map<std::string, Type::NumericValue> &paramValues,
                    const std::map<std::string, InitVarSnippet::Init> &varValues,
                    const std::map<std::string, Models::WUVarReference> &varRefTargets,
                    const std::map<std::string, Models::EGPReference> &egpRefTarget,
                    const std::string &description) const
{
     // Superclass
    Snippet::Base::validate(paramValues, description);

    // Validate variables
    const auto vars = getVars();
    Utils::validateVecNames(vars, "Variable");
    Utils::validateInitialisers(vars, varValues, "variable", description);

    // Validate variable references
    const auto varRefs = getVarRefs();
    Utils::validateVecNames(varRefs, "Variable reference");
    Utils::validateInitialisers(varRefs, varRefTargets, "Variable reference", description);
    
    // Validate EGP references
    const auto egpRefs = getExtraGlobalParamRefs();
    Utils::validateVecNames(egpRefs, "Extra global parameter reference");
    Utils::validateInitialisers(egpRefs, egpRefTarget, "Extra Global Parameter reference", description);
}
}   // namespace GeNN::CustomUpdateModels