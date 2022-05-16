#include "initVarSnippet.h"

// Implement value initialization snippets
IMPLEMENT_SNIPPET(InitVarSnippet::Uninitialised);
IMPLEMENT_SNIPPET(InitVarSnippet::Constant);
IMPLEMENT_SNIPPET(InitVarSnippet::Kernel);
IMPLEMENT_SNIPPET(InitVarSnippet::Uniform);
IMPLEMENT_SNIPPET(InitVarSnippet::Normal);
IMPLEMENT_SNIPPET(InitVarSnippet::NormalClipped);
IMPLEMENT_SNIPPET(InitVarSnippet::NormalClippedDelay);
IMPLEMENT_SNIPPET(InitVarSnippet::Exponential);
IMPLEMENT_SNIPPET(InitVarSnippet::Gamma);
IMPLEMENT_SNIPPET(InitVarSnippet::Binomial);

//----------------------------------------------------------------------------
// InitVarSnippet::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type InitVarSnippet::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void InitVarSnippet::Base::validate(const std::unordered_map<std::string, double> &paramValues) const
{
    // Superclass
    Snippet::Base::validate(paramValues, "Variable initialiser ");
}
//----------------------------------------------------------------------------
bool InitVarSnippet::Base::requiresKernel() const
{
    return (getCode().find("$(id_kernel)") != std::string::npos);
}

