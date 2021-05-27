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
void CustomUpdateModels::Base::updateHash(boost::uuids::detail::sha1 &hash) const
{
    // Superclass
    Models::Base::updateHash(hash);

    Utils::updateHash(getUpdateCode(), hash);
    Utils::updateHash(getVarRefs(), hash);
}
