#include "customConnectivityUpdateModels.h"

//----------------------------------------------------------------------------
// CustomConnectivityUpdateModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomConnectivityUpdateModels::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getRowUpdateCode(), hash);
    Utils::updateHash(getColUpdateCode(), hash);
    Utils::updateHash(getHostUpdateCode(), hash);

    Utils::updateHash(getPreVars(), hash);
    Utils::updateHash(getPostVars(), hash);

    Utils::updateHash(getWUVarRefs(), hash);
    Utils::updateHash(getPreVarRefs(), hash);
    Utils::updateHash(getPostVarRefs(), hash);
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomConnectivityUpdateModels::Base::validate() const
{
    // Superclass
    Models::Base::validate();

    Utils::validateVecNames(getPreVars(), "Presynaptic variable");
    Utils::validateVecNames(getPostVars(), "Presynaptic variable");
    Utils::validateVecNames(getWUVarRefs(), "WU variable reference");
    Utils::validateVecNames(getPreVarRefs(), "Presynaptic variable reference");
    Utils::validateVecNames(getPostVarRefs(), "Postsynaptic variable reference");

    // If any variables have a reduction access mode, give an error
    // **YUCK** copy-paste from WUM - could go in helper/Models::Base
    const auto vars = getVars();
    const auto preVars = getPreVars();
    const auto postVars = getPostVars();
    if (std::any_of(vars.cbegin(), vars.cend(),
                    [](const Models::Base::Var &v) { return (v.access & VarAccessModeAttribute::REDUCE); })
        || std::any_of(preVars.cbegin(), preVars.cend(),
                       [](const Models::Base::Var &v) { return (v.access & VarAccessModeAttribute::REDUCE); })
        || std::any_of(postVars.cbegin(), postVars.cend(),
                       [](const Models::Base::Var &v) { return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        throw std::runtime_error("Weight update models cannot include variables with REDUCE access modes - they are only supported by custom update models");
    }

    // If any variables have shared neuron duplication mode, give an error
    // **YUCK** copy-paste from WUM - could go in helper/Models::Base
    if (std::any_of(vars.cbegin(), vars.cend(),
                    [](const Models::Base::Var &v) { return (v.access & VarAccessDuplication::SHARED_NEURON); }))
    {
        throw std::runtime_error("Weight update models cannot include variables with SHARED_NEURON access modes - they are only supported on pre, postsynaptic or neuron variables");
    }
}
