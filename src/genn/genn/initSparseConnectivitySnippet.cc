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
// updateHash overrides
//----------------------------------------------------------------------------
void InitSparseConnectivitySnippet::updateHash(const Base &c, boost::uuids::detail::sha1 &hash)
{
    // Superclass
    Snippet::updateHash(c, hash);

    Utils::updateHash(c.getRowBuildCode(), hash);
    Utils::updateHash(c.getRowBuildStateVars(), hash);
    Utils::updateHash(c.getColBuildCode(), hash);
    Utils::updateHash(c.getColBuildStateVars(), hash);
    Utils::updateHash(c.getHostInitCode(), hash);
}
