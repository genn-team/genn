#include "initVarSnippet.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::InitVarSnippet
{
// Implement value initialization snippets
IMPLEMENT_SNIPPET(Uninitialised);
IMPLEMENT_SNIPPET(Constant);
IMPLEMENT_SNIPPET(Kernel);
IMPLEMENT_SNIPPET(Uniform);
IMPLEMENT_SNIPPET(Normal);
IMPLEMENT_SNIPPET(NormalClipped);
IMPLEMENT_SNIPPET(NormalClippedDelay);
IMPLEMENT_SNIPPET(Exponential);
IMPLEMENT_SNIPPET(Gamma);
IMPLEMENT_SNIPPET(Binomial);

//----------------------------------------------------------------------------
// GeNN::InitVarSnippet::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::map<std::string, Type::NumericValue> &paramValues) const
{
    // Superclass
    Snippet::Base::validate(paramValues, "Variable initialiser ");
}


//----------------------------------------------------------------------------
// Init
//----------------------------------------------------------------------------
Init::Init(const Base *snippet, const std::map<std::string, Type::NumericValue> &params)
:   Snippet::Init<Base>(snippet, params)
{
    // Validate
    getSnippet()->validate(getParams());

    // Scan code tokens
    m_CodeTokens = Utils::scanCode(getSnippet()->getCode(), "Variable initialisation code");
}
//----------------------------------------------------------------------------
Init::Init(double constant)
:   Snippet::Init<Base>(Constant::getInstance(), {{"constant", constant}})
{
    // Validate
    getSnippet()->validate(getParams());

    // Scan code tokens
    m_CodeTokens = Utils::scanCode(getSnippet()->getCode(), "Variable initialisation code");
}
//----------------------------------------------------------------------------
bool Init::isRNGRequired() const
{
    return Utils::isRNGRequired(m_CodeTokens);
}
//----------------------------------------------------------------------------
bool Init::isKernelRequired() const
{
    return Utils::isIdentifierReferenced("id_kernel", m_CodeTokens);
}
}   // namespace GeNN::InitVarSnippet