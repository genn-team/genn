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
