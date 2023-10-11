#include "customConnectivityUpdateModels.h"

// GeNN includes
#include "gennUtils.h"

//----------------------------------------------------------------------------
// GeNN::CustomConnectivityUpdateModels::Base
//----------------------------------------------------------------------------
namespace GeNN::CustomConnectivityUpdateModels
{
boost::uuids::detail::sha1::digest_type Base::getHashDigest() const
{
    // Superclass
    boost::uuids::detail::sha1 hash;
    Models::Base::updateHash(hash);

    Utils::updateHash(getRowUpdateCode(), hash);
    Utils::updateHash(getHostUpdateCode(), hash);

    Utils::updateHash(getVars(), hash);
    Utils::updateHash(getVarRefs(), hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::unordered_map<std::string, Models::VarReference> &varRefTargets,
                    const std::string &description) const
{
    // Superclass
    Models::Base::validate(paramValues, description);

    const auto vars = getVars();
    const auto preVars = getPreVars();
    const auto postVars = getPostVars();
    Utils::validateVecNames(vars, "Variable");
    Utils::validateVecNames(preVars, "Presynaptic variable");
    Utils::validateVecNames(postVars, "Presynaptic variable");
    Utils::validateVecNames(getVarRefs(), "Synapse variable reference");
    
    // Validate variable initialisers
    Utils::validateInitialisers(vars, varValues, "variable", description);
    
    // Validate variable reference initialisers
    Utils::validateInitialisers(getVarRefs(), varRefTargets, "variable reference", description);
}
//----------------------------------------------------------------------------
std::vector<Base::SynapseVar> Base::getFilteredVars(bool pre, bool post) const
{
    // Copy variables into new vector if pre and post dimensions match
    std::vector<Base::SynapseVar> filteredVars;
    const auto vars = getVars();
    std::copy_if(vars.cbegin(), vars.cend(), std::back_inserter(filteredVars),
                 [pre, post](const auto &v)
                 {
                     const auto dim = getVarAccessDim(v.access);
                     return (((dim & VarAccessDim::PRE_NEURON) == pre) 
                             && ((dim & VarAccessDim::POST_NEURON) == post));
                 });
    return filteredVars;
}
}   // namespace GeNN::CustomConnectivityUpdateModels