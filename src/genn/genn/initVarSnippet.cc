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

//----------------------------------------------------------------------------
// InitVarSnippet::Base
//----------------------------------------------------------------------------
bool InitVarSnippet::Base::canBeMerged(const Base *other) const
{
    return (Snippet::Base::canBeMerged(other)
            && (getCode() == other->getCode()));
}
//----------------------------------------------------------------------------
bool InitVarSnippet::Base::requiresKernel() const
{
    return (getCode().find("$(id_kernel)") != std::string::npos);
}

//----------------------------------------------------------------------------
// updateHash overrides
//----------------------------------------------------------------------------
void InitVarSnippet::updateHash(const Base &v, boost::uuids::detail::sha1 &hash)
{
    // Superclass
    Snippet::updateHash(v, hash);

    Utils::updateHash(v.getCode(), hash);
}
