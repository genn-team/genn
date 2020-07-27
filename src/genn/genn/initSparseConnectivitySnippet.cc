#include "initSparseConnectivitySnippet.h"

// Implement sparse connectivity initialization snippets
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::Uninitialised);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::OneToOne);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedProbability);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedProbabilityNoAutapse);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedNumberPostWithReplacement);
IMPLEMENT_SNIPPET(InitSparseConnectivitySnippet::FixedNumberTotalWithReplacement);

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
