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
    const auto varRefs = getVarRefs();
    Utils::validateVecNames(vars, "Variable");
    Utils::validateVecNames(varRefs, "Variable reference");
    
    // Validate variable initialisers
    Utils::validateInitialisers(vars, varValues, "variable", description);
    
    // Validate variable reference initialisers
    Utils::validateInitialisers(varRefs, varRefTargets, "variable reference", description);
}
//----------------------------------------------------------------------------
std::vector<Base::CustomConnectivityUpdateVar> Base::getFilteredVars(bool pre, bool post) const
{
    // Copy variables into new vector if pre and post dimensions match
    std::vector<Base::CustomConnectivityUpdateVar> filteredVars;
    const auto vars = getVars();
    std::copy_if(vars.cbegin(), vars.cend(), std::back_inserter(filteredVars),
                 [pre, post](const auto &v)
                 {
                     // **NOTE** these are defined subtractively so we're 
                     // 1) Manually extracting the dimensions that are being subtracted 
                     // 2) Checking that the dimensions we want AREN'T present in these
                     const auto dim = static_cast<VarAccessDim>(static_cast<unsigned int>(v.access) & ~0x1F);
                     return (((dim & VarAccessDim::PRE_NEURON) != pre) 
                             && ((dim & VarAccessDim::POST_NEURON) != post));
                 });
    return filteredVars;
}
}   // namespace GeNN::CustomConnectivityUpdateModels