#include "customUpdateModels.h"

// Implement models
IMPLEMENT_SNIPPET(CustomUpdateModels::Transpose);

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
bool CustomUpdateModels::Base::isReduction() const
{
    // Return true if any variables or variable references have REDUCE flag in their access mode
    const auto vars = getVars();
    const auto varRefs = getVarRefs();
    return (std::any_of(vars.cbegin(), vars.cend(),
                        [](const Models::Base::Var &v) { return (v.access & VarAccessModeAttribute::REDUCE); })
            || std::any_of(varRefs.cbegin(), varRefs.cend(),
                           [](const Models::Base::VarRef &v) { return (v.access & VarAccessModeAttribute::REDUCE); }));
}