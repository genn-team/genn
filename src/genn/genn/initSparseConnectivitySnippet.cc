#include "initSparseConnectivitySnippet.h"

// Implement sparse connectivity initialization snippets
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::Uninitialised);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::OneToOne);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedProbability);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedProbabilityNoAutapse);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedNumberPostWithReplacement);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedNumberPreWithReplacement);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::Conv2D);

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type InitSparseConnectivitySnippet::Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getRowBuildCode(), hash);
    Utils::updateHash(getRowBuildStateVars(), hash);
    Utils::updateHash(getColBuildCode(), hash);
    Utils::updateHash(getColBuildStateVars(), hash);
    Utils::updateHash(getHostInitCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void InitSparseConnectivitySnippet::Base::validateNames() const
{
    // Superclass
    Snippet::Base::validateNames();
    Utils::validateVecNames(getRowBuildStateVars(), "Row building state variable");
    Utils::validateVecNames(getColBuildStateVars(), "Column building state variable");
}
