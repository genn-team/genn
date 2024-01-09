#include "initSparseConnectivitySnippet.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::InitSparseConnectivitySnippet
{
// Implement sparse connectivity initialization snippets
IMPLEMENT_SNIPPET(Uninitialised);
IMPLEMENT_SNIPPET(OneToOne);
IMPLEMENT_SNIPPET(FixedProbability);
IMPLEMENT_SNIPPET(FixedProbabilityNoAutapse);
IMPLEMENT_SNIPPET(FixedNumberPostWithReplacement);
IMPLEMENT_SNIPPET(FixedNumberTotalWithReplacement);
IMPLEMENT_SNIPPET(FixedNumberPreWithReplacement);
IMPLEMENT_SNIPPET(Conv2D);

//----------------------------------------------------------------------------
// GeNN::InitSparseConnectivitySnippet::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getRowBuildCode(), hash);
    Utils::updateHash(getColBuildCode(), hash);
    Utils::updateHash(getHostInitCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues) const
{
    // Superclass
    Snippet::Base::validate(paramValues, "Sparse connectivity initialiser ");
}

//----------------------------------------------------------------------------
// GeNN::InitSparseConnectivitySnippet::Init
//----------------------------------------------------------------------------
Init::Init(const Base *snippet, const std::unordered_map<std::string, Type::NumericValue> &params)
:   Snippet::Init<Base>(snippet, params)
{
    // Validate
    getSnippet()->validate(getParams());

    // Scan code tokens
    m_RowBuildCodeTokens = Utils::scanCode(getSnippet()->getRowBuildCode(), "Sparse connectivity row build code");
    m_ColBuildCodeTokens = Utils::scanCode(getSnippet()->getColBuildCode(), "Sparse connectivity col build code");
    m_HostInitCodeTokens = Utils::scanCode(getSnippet()->getHostInitCode(), "Sparse connectivity host init code");
}
//----------------------------------------------------------------------------
bool Init::isRNGRequired() const
{
    return (Utils::isRNGRequired(m_RowBuildCodeTokens) || Utils::isRNGRequired(m_ColBuildCodeTokens));
}
//----------------------------------------------------------------------------
bool Init::isHostRNGRequired() const
{
    return Utils::isRNGRequired(m_HostInitCodeTokens);
}
}   // namespace GeNN::InitSparseConnectivitySnippet
