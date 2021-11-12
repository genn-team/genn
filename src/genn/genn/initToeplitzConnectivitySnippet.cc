#include "initToeplitzConnectivitySnippet.h"

// Implement sparse connectivity initialization snippets
IMPLEMENT_SNIPPET(InitToeplitzConnectivitySnippet::Uninitialised);
IMPLEMENT_SNIPPET(InitToeplitzConnectivitySnippet::Conv2D);

//----------------------------------------------------------------------------
// InitToeplitzConnectivitySnippet::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type InitToeplitzConnectivitySnippet::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getDiagonalBuildCode(), hash);
    Utils::updateHash(getDiagonalBuildStateVars(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void InitToeplitzConnectivitySnippet::Base::validate() const
{
    // Superclass
    Snippet::Base::validate();
    Utils::validateVecNames(getDiagonalBuildStateVars(), "Row building state variable");
}
