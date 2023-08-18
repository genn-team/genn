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

    Utils::updateHash(getPreVars(), hash);
    Utils::updateHash(getPostVars(), hash);

    Utils::updateHash(getVarRefs(), hash);
    Utils::updateHash(getPreVarRefs(), hash);
    Utils::updateHash(getPostVarRefs(), hash);
    
    return hash.get_digest();
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::unordered_map<std::string, InitVarSnippet::Init> &preVarValues,
                    const std::unordered_map<std::string, InitVarSnippet::Init> &postVarValues,
                    const std::unordered_map<std::string, Models::WUVarReference> &varRefTargets,
                    const std::unordered_map<std::string, Models::VarReference> &preVarRefTargets,
                    const std::unordered_map<std::string, Models::VarReference> &postVarRefTargets,
                    const std::string &description) const
{
    // Superclass
    Models::Base::validate(paramValues, varValues, description);

    const auto vars = getVars();
    const auto preVars = getPreVars();
    const auto postVars = getPostVars();
    Utils::validateVecNames(preVars, "Presynaptic variable");
    Utils::validateVecNames(postVars, "Presynaptic variable");
    Utils::validateVecNames(getVarRefs(), "Synapse variable reference");
    Utils::validateVecNames(getPreVarRefs(), "Presynaptic variable reference");
    Utils::validateVecNames(getPostVarRefs(), "Postsynaptic variable reference");
    
    // Validate variable initialisers
    Utils::validateInitialisers(preVars, preVarValues, "presynaptic variable", description);
    Utils::validateInitialisers(postVars, postVarValues, "postsynaptic variable", description);
    
    // Validate variable reference initialisers
    Utils::validateInitialisers(getVarRefs(), varRefTargets, "variable reference", description);
    Utils::validateInitialisers(getPreVarRefs(), preVarRefTargets, "presynaptic variable reference", description);
    Utils::validateInitialisers(getPostVarRefs(), postVarRefTargets, "postsynaptic variable reference", description);
    
    // Check variables have suitable access types
    if(std::any_of(vars.cbegin(), vars.cend(),
                   [](const Models::Base::Var &v){ return !v.access.template isValid<SynapseVarAccess>(); }))
    {
        throw std::runtime_error("Custom connectivity update models variables must have SynapseVarAccess access type");
    }
    if(std::any_of(preVars.cbegin(), preVars.cend(),
                   [](const Models::Base::Var &v){ return !v.access.template isValid<NeuronVarAccess>(); }))
    {
        throw std::runtime_error("Custom connectivity update models presynaptic variables must have NeuronVarAccess access type");
    }
    if(std::any_of(postVars.cbegin(), postVars.cend(),
                   [](const Models::Base::Var &v){ return !v.access.template isValid<NeuronVarAccess>(); }))
    {
        throw std::runtime_error("Custom connectivity update models postsynaptic variables must have NeuronVarAccess access type");
    }
}
}   // namespace GeNN::CustomConnectivityUpdateModels