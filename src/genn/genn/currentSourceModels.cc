#include "currentSourceModels.h"

// Implement models
IMPLEMENT_MODEL(CurrentSourceModels::DC);
IMPLEMENT_MODEL(CurrentSourceModels::GaussianNoise);
IMPLEMENT_MODEL(CurrentSourceModels::PoissonExp);

//----------------------------------------------------------------------------
// CurrentSourceModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CurrentSourceModels::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getInjectionCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CurrentSourceModels::Base::validate() const
{
    // Superclass
    Models::Base::validate();

    // If any variables have a reduction access mode, give an error
    const auto vars = getVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                   [](const Models::Base::Var &v){ return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        throw std::runtime_error("Current source models cannot include variables with REDUCE access modes - they are only supported by custom update models");
    }
}
