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
bool InitSparseConnectivitySnippet::Base::canBeMerged(const Base *other) const
{
    return (Snippet::Base::canBeMerged(other)
            && (getRowBuildCode() == other->getRowBuildCode())
            && (getRowBuildStateVars() == other->getRowBuildStateVars())
            && (getColBuildCode() == other->getColBuildCode())
            && (getColBuildStateVars() == other->getColBuildStateVars())
            && (getHostInitCode() == other->getHostInitCode()));
}
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
