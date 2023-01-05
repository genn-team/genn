#include "initToeplitzConnectivitySnippet.h"

using namespace GeNN;

namespace GeNN::InitToeplitzConnectivitySnippet
{
// Implement sparse connectivity initialization snippets
IMPLEMENT_SNIPPET(Uninitialised);
IMPLEMENT_SNIPPET(Conv2D);
IMPLEMENT_SNIPPET(AvgPoolConv2D);

//----------------------------------------------------------------------------
// GeNN::InitToeplitzConnectivitySnippet::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getDiagonalBuildCode(), hash);
    Utils::updateHash(getDiagonalBuildStateVars(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues) const
{
    // Superclass
    Snippet::Base::validate(paramValues, "Toeplitz connectivity initialiser ");
    Utils::validateVecNames(getDiagonalBuildStateVars(), "Row building state variable");
}
}   // namespace GeNN::InitToeplitzConnectivitySnippet