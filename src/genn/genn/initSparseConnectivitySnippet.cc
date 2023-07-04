#include "initSparseConnectivitySnippet.h"

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
    Utils::updateHash(getRowBuildStateVars(), hash);
    Utils::updateHash(getColBuildCode(), hash);
    Utils::updateHash(getColBuildStateVars(), hash);
    Utils::updateHash(getHostInitCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues) const
{
    // Superclass
    Snippet::Base::validate(paramValues, "Sparse connectivity initialiser ");
    Utils::validateVecNames(getRowBuildStateVars(), "Row building state variable");
    Utils::validateVecNames(getColBuildStateVars(), "Column building state variable");
}

//----------------------------------------------------------------------------
// GeNN::InitSparseConnectivitySnippet::Init
//----------------------------------------------------------------------------
void Init::finalise(double dt, const Type::TypeContext &context, const std::string &errorContext)
{
    // Superclass
    Snippet::Init<Base>::finalise(dt);

    // Scan code tokens
    m_RowBuildCodeTokens = Utils::scanCode(getSnippet()->getRowBuildCode(), context, errorContext + "row build code");
    m_ColBuildCodeTokens = Utils::scanCode(getSnippet()->getColBuildCode(), context, errorContext + "col build code");
    m_HostInitCodeTokens = Utils::scanCode(getSnippet()->getHostInitCode(), context, errorContext + "host init code");
}
//----------------------------------------------------------------------------
bool Init::isRNGRequired() const
{
    return (Utils::isRNGRequired(m_RowBuildCodeTokens) || Utils::isRNGRequired(m_ColBuildCodeTokens));
}
//----------------------------------------------------------------------------
bool Init::isHostRNGRequired() const
{
    return Utils::isRNGRequired(m_HostInitTokens);
}
}   // namespace GeNN::InitSparseConnectivitySnippet
