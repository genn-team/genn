#include "customUpdateModels.h"


// Implement models
IMPLEMENT_MODEL(CustomUpdateModels::Transpose);

//----------------------------------------------------------------------------
// CustomUpdateModels::Base
//----------------------------------------------------------------------------
bool CustomUpdateModels::Base::canBeMerged(const Base *other) const
{
    return (Models::Base::canBeMerged(other)
            && (getUpdateCode() == other->getUpdateCode())
            && (getVarRefs() == other->getVarRefs()));
}

//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateModels::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getUpdateCode(), hash);
    Utils::updateHash(getVarRefs(), hash);
    return hash.get_digest();
}
