#include "customUpdateModels.h"


// Implement models
IMPLEMENT_MODEL(CustomUpdateModels::Transpose);

//----------------------------------------------------------------------------
// CustomUpdateModels::Base
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
//----------------------------------------------------------------------------
void CustomUpdateModels::Base::validateNames() const
{
    // Superclass
    Models::Base::validateNames();

    Utils::validateVecNames(getVarRefs(), "Variable reference");
}
