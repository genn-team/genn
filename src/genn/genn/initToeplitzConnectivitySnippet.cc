#include "initToeplitzConnectivitySnippet.h"

// GeNN includes
#include "gennUtils.h"

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

//----------------------------------------------------------------------------
// GeNN::InitToeplitzConnectivitySnippet::Init
//----------------------------------------------------------------------------
Init::Init(const Base *snippet, const std::unordered_map<std::string, double> &params)
:   Snippet::Init<Base>(snippet, params)
{
    // Scan code tokens
    m_DiagonalBuildCodeTokens = Utils::scanCode(getSnippet()->getDiagonalBuildCode(), "Toeplitz connectivity diagonal build code");
}
//----------------------------------------------------------------------------
bool Init::isRNGRequired() const
{
    return Utils::isRNGRequired(m_DiagonalBuildCodeTokens);
}
}   // namespace GeNN::InitToeplitzConnectivitySnippet