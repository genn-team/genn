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
    Utils::updateHash(getExtraGlobalParamRefs(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomUpdateModels::Base::validate() const
{
    // Superclass
    Models::Base::validate();

    const auto egpRefs = getExtraGlobalParamRefs();
    Utils::validateVecNames(getVarRefs(), "Variable reference");
    Utils::validateVecNames(egpRefs, "Extra global parameter reference");

    // If any EGP references have non-pointer type, give error
    if (std::any_of(egpRefs.cbegin(), egpRefs.cend(),
                    [](const Models::Base::EGPRef &e) { return !Utils::isTypePointer(e.type); }))
    {
        throw std::runtime_error("Extra global parameter references can only be used with pointer EGPs");
    }
}
