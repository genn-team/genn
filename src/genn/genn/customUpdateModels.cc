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
// updateHash overrides
//----------------------------------------------------------------------------
void CustomUpdateModels::updateHash(const Base &c, boost::uuids::detail::sha1 &hash)
{
    // Superclass
    Models::updateHash(c, hash);

    Utils::updateHash(c.getUpdateCode(), hash);
    Utils::updateHash(c.getVarRefs(), hash);
}
