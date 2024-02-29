#include "neuronModels.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::NeuronModels
{
// Implement models
IMPLEMENT_SNIPPET(RulkovMap);
IMPLEMENT_SNIPPET(Izhikevich);
IMPLEMENT_SNIPPET(IzhikevichVariable);
IMPLEMENT_SNIPPET(LIF);
IMPLEMENT_SNIPPET(SpikeSource);
IMPLEMENT_SNIPPET(SpikeSourceArray);
IMPLEMENT_SNIPPET(Poisson);
IMPLEMENT_SNIPPET(PoissonNew);
IMPLEMENT_SNIPPET(TraubMiles);
IMPLEMENT_SNIPPET(TraubMilesFast);
IMPLEMENT_SNIPPET(TraubMilesAlt);
IMPLEMENT_SNIPPET(TraubMilesNStep);

//----------------------------------------------------------------------------
// GeNN::NeuronModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);
    Utils::updateHash(getVars(), hash);
    Utils::updateHash(getSimCode(), hash);
    Utils::updateHash(getThresholdConditionCode(), hash);
    Utils::updateHash(getResetCode(), hash);
    Utils::updateHash(isAutoRefractoryRequired(), hash);
    Utils::updateHash(getAdditionalInputVars(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::map<std::string, Type::NumericValue> &paramValues, 
                    const std::map<std::string, InitVarSnippet::Init> &varValues,
                    const std::string &description) const
{
    // Superclass
    Models::Base::validate(paramValues, description);

    Utils::validateVecNames(getAdditionalInputVars(), "Additional input variable");

    // Validate variable names
    const auto vars = getVars();
    Utils::validateVecNames(vars, "Variable");

    // Validate variable initialisers
    Utils::validateInitialisers(vars, varValues, "variable", description);
}
}   // namespace GeNN::NeuronModels
