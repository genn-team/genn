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
    Utils::updateHash(getEventCode(), hash);
    Utils::updateHash(getLearnPostCode(), hash);
    Utils::updateHash(getSynapseDynamicsCode(), hash);
    Utils::updateHash(getEventThresholdConditionCode(), hash);
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
void Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::unordered_map<std::string, InitVarSnippet::Init> &preVarValues,
                    const std::unordered_map<std::string, InitVarSnippet::Init> &postVarValues,
                    const std::unordered_map<std::string, Models::VarReference> &preVarRefTargets,
                    const std::unordered_map<std::string, Models::VarReference> &postVarRefTargets,
                    const std::string &description) const
{
    // Superclass
    Snippet::Base::validate(paramValues, description);

    const auto vars = getVars();
    const auto preVars = getPreVars();
    const auto postVars = getPostVars();
    Utils::validateVecNames(getVars(), "Variable");
    Utils::validateVecNames(getPreVars(), "Presynaptic variable");
    Utils::validateVecNames(getPostVars(), "Presynaptic variable");

    // Validate variable initialisers
    Utils::validateInitialisers(vars, varValues, "variable", description);
    Utils::validateInitialisers(preVars, preVarValues, "presynaptic variable", description);
    Utils::validateInitialisers(postVars, postVarValues, "postsynaptic variable", description);

    // Validate variable reference initialisers
    const auto preVarRefs = getPreNeuronVarRefs();
    const auto postVarRefs = getPostNeuronVarRefs();
    Utils::validateVecNames(preVarRefs, "Presynaptic neuron variable reference");
    Utils::validateVecNames(postVarRefs, "Postsynaptic neuron variable reference");
    Utils::validateInitialisers(preVarRefs, preVarRefTargets, "Presynaptic neuron variable reference", description);
    Utils::validateInitialisers(postVarRefs, postVarRefTargets, "Postsyanptic neuron variable reference", description);
}
}   // namespace WeightUpdateModels