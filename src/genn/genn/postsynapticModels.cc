#include "postsynapticModels.h"

// Implement models
IMPLEMENT_MODEL(PostsynapticModels::ExpCurr);
IMPLEMENT_MODEL(PostsynapticModels::ExpCond);
IMPLEMENT_MODEL(PostsynapticModels::DeltaCurr);


//----------------------------------------------------------------------------
// PostsynapticModels::Base
//----------------------------------------------------------------------------
bool PostsynapticModels::Base::canBeMerged(const Base *other) const
{
    return (Models::Base::canBeMerged(other)
            && (getDecayCode() == other->getDecayCode())
            && (getApplyInputCode() == other->getApplyInputCode())
            && (getSupportCode() == other->getSupportCode()));
}


//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
void PostsynapticModels::updateHash(const Base &p, boost::uuids::detail::sha1 &hash)
{
    // Superclass
    Models::updateHash(p, hash);

    Utils::updateHash(p.getDecayCode(), hash);
    Utils::updateHash(p.getApplyInputCode(), hash);
    Utils::updateHash(p.getSupportCode(), hash);
}
