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
boost::uuids::detail::sha1::digest_type PostsynapticModels::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getDecayCode(), hash);
    Utils::updateHash(getApplyInputCode(), hash);
    Utils::updateHash(getSupportCode(), hash);
    return hash.get_digest();
}
