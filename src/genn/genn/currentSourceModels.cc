#include "currentSourceModels.h"

// GeNN includes
#include "gennUtils.h"

using namespace GeNN;

namespace GeNN::CurrentSourceModels
{
// Implement models
IMPLEMENT_SNIPPET(DC);
IMPLEMENT_SNIPPET(GaussianNoise);
IMPLEMENT_SNIPPET(PoissonExp);

//----------------------------------------------------------------------------
// GeNN::CurrentSourceModels::Base
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getVars(), hash);
    Utils::updateHash(getNeuronVarRefs(), hash);
    Utils::updateHash(getInjectionCode(), hash);
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, Type::NumericValue> &paramValues, 
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::unordered_map<std::string, Models::VarReference> &varRefTargets,
                    const std::string &description) const
{
    // Superclass
    Snippet::Base::validate(paramValues, description);

    // Validate variable names
    const auto vars = getVars();
    Utils::validateVecNames(vars, "Variable");

    // Validate variable initialisers
    Utils::validateInitialisers(vars, varValues, "variable", description);

    // Validate variable reference initialisers
    const auto varRefs = getNeuronVarRefs();
    Utils::validateVecNames(varRefs, "Neuron variable reference");
    Utils::validateInitialisers(varRefs, varRefTargets, "Neuron variable reference", description);
}
}   // namespace GeNN::CurrentSourceModels