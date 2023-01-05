#include "initVarSnippet.h"

using namespace GeNN;

namespace GeNN::InitVarSnippet
{
// Implement value initialization snippets
IMPLEMENT_SNIPPET(Uninitialised);
IMPLEMENT_SNIPPET(Constant);
IMPLEMENT_SNIPPET(Kernel);
IMPLEMENT_SNIPPET(Uniform);
IMPLEMENT_SNIPPET(Normal);
IMPLEMENT_SNIPPET(NormalClipped);
IMPLEMENT_SNIPPET(NormalClippedDelay);
IMPLEMENT_SNIPPET(Exponential);
IMPLEMENT_SNIPPET(Gamma);
IMPLEMENT_SNIPPET(Binomial);

//----------------------------------------------------------------------------
// GeNN::InitVarSnippet::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues) const
{
    // Superclass
    Snippet::Base::validate(paramValues, "Variable initialiser ");
}
//----------------------------------------------------------------------------
bool Base::requiresKernel() const
{
    return (getCode().find("$(id_kernel)") != std::string::npos);
}
}   // namespace GeNN::InitVarSnippet