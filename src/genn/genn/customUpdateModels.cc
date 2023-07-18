#include "customUpdateModels.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::CustomUpdateModels
{
// Implement models
IMPLEMENT_SNIPPET(Transpose);

//----------------------------------------------------------------------------
// GeNN::CustomUpdateModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getUpdateCode(), hash);
    Utils::updateHash(getVarRefs(), hash);
    return hash.get_digest();
}
}   // namespace GeNN::CustomUpdateModels