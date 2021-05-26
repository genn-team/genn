#include "currentSourceModels.h"

// Implement models
IMPLEMENT_MODEL(CurrentSourceModels::DC);
IMPLEMENT_MODEL(CurrentSourceModels::GaussianNoise);
IMPLEMENT_MODEL(CurrentSourceModels::PoissonExp);

//----------------------------------------------------------------------------
// CurrentSourceModels::Base
//----------------------------------------------------------------------------
bool CurrentSourceModels::Base::canBeMerged(const Base *other) const
{
    return (Models::Base::canBeMerged(other)
            && (getInjectionCode() == other->getInjectionCode()));
}

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
void CurrentSourceModels::updateHash(const Base &c, boost::uuids::detail::sha1 &hash)
{
    // Superclass
    Models::updateHash(c, hash);

    Utils::updateHash(c.getInjectionCode(), hash);
}
