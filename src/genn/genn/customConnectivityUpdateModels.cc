#include "customConnectivityUpdateModels.h"

//----------------------------------------------------------------------------
// GeNN::CustomConnectivityUpdateModels::Base
//----------------------------------------------------------------------------
namespace GeNN::CustomConnectivityUpdateModels
{
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getRowUpdateCode(), hash);
    Utils::updateHash(getHostUpdateCode(), hash);

    Utils::updateHash(getPreVars(), hash);
    Utils::updateHash(getPostVars(), hash);

    Utils::updateHash(getVarRefs(), hash);
    Utils::updateHash(getPreVarRefs(), hash);
    Utils::updateHash(getPostVarRefs(), hash);
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                    const std::unordered_map<std::string, Models::VarInit> &varValues,
                    const std::unordered_map<std::string, Models::VarInit> &preVarValues,
                    const std::unordered_map<std::string, Models::VarInit> &postVarValues,
                    const std::unordered_map<std::string, Models::WUVarReference> &varRefTargets,
                    const std::unordered_map<std::string, Models::VarReference> &preVarRefTargets,
                    const std::unordered_map<std::string, Models::VarReference> &postVarRefTargets,
                    const std::string &description) const
{
    // Superclass
    Models::Base::validate(paramValues, varValues, description);

    const auto vars = getVars();
    const auto preVars = getPreVars();
    const auto postVars = getPostVars();
    Utils::validateVecNames(preVars, "Presynaptic variable");
    Utils::validateVecNames(postVars, "Presynaptic variable");
    Utils::validateVecNames(getVarRefs(), "Synapse variable reference");
    Utils::validateVecNames(getPreVarRefs(), "Presynaptic variable reference");
    Utils::validateVecNames(getPostVarRefs(), "Postsynaptic variable reference");

    
    // Validate variable initialisers
    Utils::validateInitialisers(preVars, preVarValues, "presynaptic variable", description);
    Utils::validateInitialisers(postVars, postVarValues, "postsynaptic variable", description);
    
    Utils::validateInitialisers(getVarRefs(), varRefTargets, "variable reference", description);
    Utils::validateInitialisers(getPreVarRefs(), preVarRefTargets, "presynaptic variable reference", description);
    Utils::validateInitialisers(getPostVarRefs(), postVarRefTargets, "postsynaptic variable reference", description);
    
    
    // If any variables have a reduction access mode, give an error
    // **YUCK** copy-paste from WUM - could go in helper/Models::Base
    if (std::any_of(vars.cbegin(), vars.cend(),
                    [](const Models::Base::Var &v) { return (v.access & VarAccessModeAttribute::REDUCE); })
        || std::any_of(preVars.cbegin(), preVars.cend(),
                       [](const Models::Base::Var &v) { return (v.access & VarAccessModeAttribute::REDUCE); })
        || std::any_of(postVars.cbegin(), postVars.cend(),
                       [](const Models::Base::Var &v) { return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        throw std::runtime_error("Custom connectivity update models cannot include variables with REDUCE access modes - they are only supported by custom update models");
    }

    // If any variables have shared neuron duplication mode, give an error
    // **YUCK** copy-paste from WUM - could go in helper/Models::Base
    if (std::any_of(vars.cbegin(), vars.cend(),
                    [](const Models::Base::Var &v) { return (v.access & VarAccessDuplication::SHARED_NEURON); }))
    {
        throw std::runtime_error("Custom connectivity update models cannot include variables with SHARED_NEURON access modes - they are only supported on pre, postsynaptic or neuron variables");
    }
}
}   // namespace GeNN::CustomConnectivityUpdateModels