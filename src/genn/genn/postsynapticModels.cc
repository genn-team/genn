#include "postsynapticModels.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::PostsynapticModels
{
// Implement models
IMPLEMENT_SNIPPET(ExpCurr);
IMPLEMENT_SNIPPET(ExpCond);
IMPLEMENT_SNIPPET(DeltaCurr);

//----------------------------------------------------------------------------
// GeNN::PostsynapticModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);
    Utils::updateHash(getVars(), hash);
    Utils::updateHash(getNeuronVarRefs(), hash);
    Utils::updateHash(getSimCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::map<std::string, Type::NumericValue> &paramValues, 
                    const std::map<std::string, InitVarSnippet::Init> &varValues,
                    const std::map<std::string, Models::VarReference> &varRefTargets) const
{
    // Superclass
    Snippet::Base::validate(paramValues, "Postsynaptic model");

    // Validate variable names and initialisers
    const auto vars = getVars();
    Utils::validateVecNames(vars, "Variable");
    Utils::validateInitialisers(vars, varValues, "variable", "Postsynaptic model");

    // Validate variable reference initialisers
    const auto varRefs = getNeuronVarRefs();
    Utils::validateVecNames(varRefs, "Neuron variable reference");
    Utils::validateInitialisers(varRefs, varRefTargets, "Neuron variable reference", "Postsynaptic model");
}

//----------------------------------------------------------------------------
// GeNN::PostsynapticModels::Init
//----------------------------------------------------------------------------
Init::Init(const Base *snippet, const std::map<std::string, Type::NumericValue> &params, 
           const std::map<std::string, InitVarSnippet::Init> &varInitialisers, 
           const std::map<std::string, std::variant<std::string, Models::VarReference>> &neuronVarReferences)
:   Snippet::Init<Base>(snippet, params), m_VarInitialisers(varInitialisers), m_NeuronVarReferences(neuronVarReferences)
{
    // Validate
    getSnippet()->validate(getParams(), getVarInitialisers(), getNeuronVarReferences());

    // Scan code tokens
    m_SimCodeTokens = Utils::scanCode(getSnippet()->getSimCode(), "Postsynaptic model sim code");
}
//----------------------------------------------------------------------------
bool Init::isRNGRequired() const
{
    return Utils::isRNGRequired(m_SimCodeTokens);
}
//----------------------------------------------------------------------------
bool Init::isVarInitRequired() const
{
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const auto &init)
                       { 
                           return !Utils::areTokensEmpty(init.second.getCodeTokens());
                       });
}
//----------------------------------------------------------------------------
void Init::finalise(double dt)
{
    // Superclass
    Snippet::Init<Base>::finalise(dt);

     // Initialise derived parameters for PSM variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.second.finalise(dt);
    }
}
}   // namespace GeNN::PostsynapticModels