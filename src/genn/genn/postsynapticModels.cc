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
