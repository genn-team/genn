#include "weightUpdateModels.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::WeightUpdateModels
{
IMPLEMENT_SNIPPET(StaticPulse);
IMPLEMENT_SNIPPET(StaticPulseConstantWeight);
IMPLEMENT_SNIPPET(StaticPulseDendriticDelay);
IMPLEMENT_SNIPPET(StaticGraded);
IMPLEMENT_SNIPPET(PiecewiseSTDP);

//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);
    Utils::updateHash(getVars(), hash);
    Utils::updateHash(getSimCode(), hash);
    Utils::updateHash(getPreEventCode(), hash);
    Utils::updateHash(getPostEventCode(), hash);
    Utils::updateHash(getLearnPostCode(), hash);
    Utils::updateHash(getSynapseDynamicsCode(), hash);
    Utils::updateHash(getPreEventThresholdConditionCode(), hash);
    Utils::updateHash(getPostEventThresholdConditionCode(), hash);
    Utils::updateHash(getPreSpikeCode(), hash);
    Utils::updateHash(getPostSpikeCode(), hash);
    Utils::updateHash(getPreDynamicsCode(), hash);
    Utils::updateHash(getPostDynamicsCode(), hash);
    Utils::updateHash(getPreVars(), hash);
    Utils::updateHash(getPostVars(), hash);
    Utils::updateHash(getPreNeuronVarRefs(), hash);
    Utils::updateHash(getPostNeuronVarRefs(), hash);

    // Return digest
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getPreHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getPreSpikeCode(), hash);
    Utils::updateHash(getPreDynamicsCode(), hash);
    Utils::updateHash(getPreVars(), hash);
    Utils::updateHash(getPreNeuronVarRefs(), hash);

    // Return digest
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getPostHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);
    Utils::updateHash(getPostSpikeCode(), hash);
    Utils::updateHash(getPostDynamicsCode(), hash);
    Utils::updateHash(getPostVars(), hash);
    Utils::updateHash(getPostNeuronVarRefs(), hash);

    // Return digest
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getPreEventHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getPreEventThresholdConditionCode(), hash);
    Utils::updateHash(getPreVars(), hash);
    Utils::updateHash(getPreNeuronVarRefs(), hash);

    // Return digest
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getPostEventHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getPostEventThresholdConditionCode(), hash);
    Utils::updateHash(getPostVars(), hash);
    Utils::updateHash(getPostNeuronVarRefs(), hash);

    // Return digest
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues, 
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::unordered_map<std::string, InitVarSnippet::Init> &preVarValues,
                    const std::unordered_map<std::string, InitVarSnippet::Init> &postVarValues,
                    const std::unordered_map<std::string, Models::VarReference> &preVarRefTargets,
                    const std::unordered_map<std::string, Models::VarReference> &postVarRefTargets) const
{
    // Superclass
    Snippet::Base::validate(paramValues, "Weight update model");

    const auto vars = getVars();
    const auto preVars = getPreVars();
    const auto postVars = getPostVars();
    Utils::validateVecNames(getVars(), "Variable");
    Utils::validateVecNames(getPreVars(), "Presynaptic variable");
    Utils::validateVecNames(getPostVars(), "Presynaptic variable");

    // Validate variable initialisers
    Utils::validateInitialisers(vars, varValues, "variable", "Weight update model");
    Utils::validateInitialisers(preVars, preVarValues, "presynaptic variable", "Weight update model");
    Utils::validateInitialisers(postVars, postVarValues, "postsynaptic variable", "Weight update model");

    // Validate variable reference initialisers
    const auto preVarRefs = getPreNeuronVarRefs();
    const auto postVarRefs = getPostNeuronVarRefs();
    Utils::validateVecNames(preVarRefs, "Presynaptic neuron variable reference");
    Utils::validateVecNames(postVarRefs, "Postsynaptic neuron variable reference");
    Utils::validateInitialisers(preVarRefs, preVarRefTargets, "Presynaptic neuron variable reference", "Weight update model");
    Utils::validateInitialisers(postVarRefs, postVarRefTargets, "Postsyanptic neuron variable reference", "Weight update model");

    // Check if event-threshold condition code is provided, then event-handler is also provided
    if(getPreEventCode().empty() != getPreEventThresholdConditionCode().empty()) {
        throw std::runtime_error("Weight update model: to handle presynaptic spike-like events, both presynaptic event threshold condition code and presynaptic event code must be specified.");
    }    
    if(getPostEventCode().empty() != getPostEventThresholdConditionCode().empty()) {
        throw std::runtime_error("Weight update model: to handle postsynaptic spike-like events, both postsynaptic event threshold condition code and postsynaptic event code must be specified.");
    }
}


//----------------------------------------------------------------------------
// GeNN::WeightUpdateModels::Init
//----------------------------------------------------------------------------
Init::Init(const Base *snippet, const std::unordered_map<std::string, Type::NumericValue> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers, 
           const std::unordered_map<std::string, InitVarSnippet::Init> &preVarInitialisers, const std::unordered_map<std::string, InitVarSnippet::Init> &postVarInitialisers,
           const std::unordered_map<std::string, Models::VarReference> &preNeuronVarReferences, const std::unordered_map<std::string, Models::VarReference> &postNeuronVarReferences)
:   Snippet::Init<Base>(snippet, params), m_VarInitialisers(varInitialisers), m_PreVarInitialisers(preVarInitialisers), m_PostVarInitialisers(postVarInitialisers), 
    m_PreNeuronVarReferences(preNeuronVarReferences), m_PostNeuronVarReferences(postNeuronVarReferences)
{
    // Validate
    getSnippet()->validate(getParams(), getVarInitialisers(), getPreVarInitialisers(), getPostVarInitialisers(),
                           getPreNeuronVarReferences(), getPostNeuronVarReferences());

    // Check variable reference types
    Models::checkVarReferenceTypes(getPreNeuronVarReferences(), getSnippet()->getPreNeuronVarRefs());
    Models::checkVarReferenceTypes(getPostNeuronVarReferences(), getSnippet()->getPostNeuronVarRefs());

    // Scan code tokens
    m_SimCodeTokens = Utils::scanCode(getSnippet()->getSimCode(), "Weight update model sim code");
    m_PreEventCodeTokens = Utils::scanCode(getSnippet()->getPreEventCode(), "Weight update model presynaptic event code");
    m_PostEventCodeTokens = Utils::scanCode(getSnippet()->getPostEventCode(), "Weight update model postsynaptic event code");
    m_PostLearnCodeTokens = Utils::scanCode(getSnippet()->getLearnPostCode(), "Weight update model learn post code");
    m_SynapseDynamicsCodeTokens = Utils::scanCode(getSnippet()->getSynapseDynamicsCode(), "Weight update model synapse dynamics code");
    m_PreEventThresholdCodeTokens = Utils::scanCode(getSnippet()->getPreEventThresholdConditionCode(), "Presynaptic weight update model event threshold code");
    m_PostEventThresholdCodeTokens = Utils::scanCode(getSnippet()->getPostEventThresholdConditionCode(), "Postsynaptic weight update model event threshold code");
    m_PreSpikeCodeTokens = Utils::scanCode(getSnippet()->getPreSpikeCode(), "Weight update model pre spike code");
    m_PostSpikeCodeTokens = Utils::scanCode(getSnippet()->getPostSpikeCode(), "Weight update model post spike code");
    m_PreDynamicsCodeTokens = Utils::scanCode(getSnippet()->getPreDynamicsCode(), "Weight update model pre dynamics code");
    m_PostDynamicsCodeTokens = Utils::scanCode(getSnippet()->getPostDynamicsCode(), "Weight update model post dynamics code");
}
//----------------------------------------------------------------------------
void Init::finalise(double dt)
{
    // Superclass
    Snippet::Init<Base>::finalise(dt);

    // Initialise derived parameters for WU variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.second.finalise(dt);
    }

    // Initialise derived parameters for WU presynaptic variable initialisers
    for(auto &v : m_PreVarInitialisers) {
        v.second.finalise(dt);
    }
    
    // Initialise derived parameters for WU postsynaptic variable initialisers
    for(auto &v : m_PostVarInitialisers) {
        v.second.finalise(dt);
    }
}
}   // namespace WeightUpdateModels