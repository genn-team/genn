#include "currentSourceModels.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::CurrentSourceModels
{
// Implement models
IMPLEMENT_SNIPPET(DC);
IMPLEMENT_SNIPPET(GaussianNoise);
IMPLEMENT_SNIPPET(PoissonExp);

//----------------------------------------------------------------------------
// GeNN::CurrentSourceModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getInjectionCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::string &description) const
{
    // Superclass
    Models::Base::validate(paramValues, varValues, description);

    // If any variables have a reduction access mode, give an error
    const auto vars = getVars();
    if(std::any_of(vars.cbegin(), vars.cend(),
                   [](const Models::Base::Var &v){ return (v.access & VarAccessModeAttribute::REDUCE); }))
    {
        throw std::runtime_error("Current source models cannot include variables with REDUCE access modes - they are only supported by custom update models");
    }
}
}   // namespace GeNN::CurrentSourceModels