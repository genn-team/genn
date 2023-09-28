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
std::vector<Base::SynapseVar> Base::getSynVars() const
{ 
    return getFilteredSynapseVars(getVars(), true, true); 
}
//----------------------------------------------------------------------------
std::vector<Base::SynapseVar> Base::getPreVars() const
{
    return getFilteredSynapseVars(getVars(), true, false); 
}
//----------------------------------------------------------------------------
std::vector<Base::SynapseVar> Base::getPostVars() const
{ 
    return getFilteredSynapseVars(getVars(), false, true); 
}
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

    // Return digest
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::string &description) const
{
    // Superclass
    Snippet::Base::validate(paramValues, description);

    // Validate variable initialisers
    const auto vars = getVars();
    Utils::validateVecNames(vars, "Variable");
    Utils::validateInitialisers(vars, varValues, "variable", description);
}
}   // namespace WeightUpdateModels