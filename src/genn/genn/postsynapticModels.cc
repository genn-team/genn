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
void PostsynapticModels::Base::updateHash(boost::uuids::detail::sha1 &hash) const
{
    // Superclass
    Models::Base::updateHash(hash);

    Utils::updateHash(getDecayCode(), hash);
    Utils::updateHash(getApplyInputCode(), hash);
    Utils::updateHash(getSupportCode(), hash);
}
