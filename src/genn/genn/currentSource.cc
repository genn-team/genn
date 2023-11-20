#include "currentSource.h"

// Standard includes
#include <algorithm>
#include <cmath>

// GeNN includes
#include "gennUtils.h"
#include "neuronGroupInternal.h"

//------------------------------------------------------------------------
// GeNN::CurrentSource
//------------------------------------------------------------------------
namespace GeNN
{
void CurrentSource::setVarLocation(const std::string &varName, VarLocation loc) 
{ 
    if(!getCurrentSourceModel()->getVar(varName)) {
        throw std::runtime_error("Unknown current source model variable '" + varName + "'");
    }
    m_VarLocation.set(varName, loc); 
}
//----------------------------------------------------------------------------
void CurrentSource::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc) 
{ 
    if(!getCurrentSourceModel()->getExtraGlobalParam(paramName)) {
        throw std::runtime_error("Unknown current source model extra global parameter '" + paramName + "'");
    }
    m_ExtraGlobalParamLocation.set(paramName, loc); 
}
//----------------------------------------------------------------------------
void CurrentSource::setParamDynamic(const std::string &paramName, bool dynamic) 
{ 
    if(!getCurrentSourceModel()->getParam(paramName)) {
        throw std::runtime_error("Unknown current source model parameter '" + paramName + "'");
    }
    m_DynamicParams.set(paramName, dynamic); 
}
//----------------------------------------------------------------------------
void CurrentSource::setTargetVar(const std::string &varName)
{
    // If varname is either 'ISyn' or name of target neuron group additional input variable, store
    const auto additionalInputVars = getTrgNeuronGroup()->getNeuronModel()->getAdditionalInputVars();
    if(varName == "Isyn" || 
       std::find_if(additionalInputVars.cbegin(), additionalInputVars.cend(), 
                    [&varName](const Models::Base::ParamVal &v){ return (v.name == varName); }) != additionalInputVars.cend())
    {
        m_TargetVar = varName;
    }
    else {
        throw std::runtime_error("Target neuron group has no input variable '" + varName + "'");
    }
}
//----------------------------------------------------------------------------
CurrentSource::CurrentSource(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                             const std::unordered_map<std::string, Type::NumericValue> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                             const std::unordered_map<std::string, Models::VarReference> &neuronVarReferences, const NeuronGroupInternal *trgNeuronGroup, 
                             VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_CurrentSourceModel(currentSourceModel), m_Params(params), m_VarInitialisers(varInitialisers),
    m_NeuronVarReferences(neuronVarReferences), m_TrgNeuronGroup(trgNeuronGroup), m_VarLocation(defaultVarLocation), 
    m_ExtraGlobalParamLocation(defaultExtraGlobalParamLocation), m_TargetVar("Isyn")
{
    // Validate names
    Utils::validatePopName(name, "Current source");
    getCurrentSourceModel()->validate(getParams(), getVarInitialisers(), getNeuronVarReferences(), "Current source " + getName());

    // Check variable reference types
    Models::checkVarReferenceTypes(getNeuronVarReferences(), getCurrentSourceModel()->getNeuronVarRefs());

    // Check additional local variable reference constraints
    Models::checkLocalVarReferences(getNeuronVarReferences(), getCurrentSourceModel()->getNeuronVarRefs(),
                                    getTrgNeuronGroup(), "Variable references to in current source can only point to target neuron group.");
    
    // Scan current source model code string
    m_InjectionCodeTokens = Utils::scanCode(getCurrentSourceModel()->getInjectionCode(), 
                                            "Current source '" + getName() + "' injection code");
}
//----------------------------------------------------------------------------
void CurrentSource::finalise(double dt)
{
    auto derivedParams = getCurrentSourceModel()->getDerivedParams();

    // Loop through derived parameters
    for(const auto &d : derivedParams) {
        m_DerivedParams.emplace(d.name, d.func(m_Params, dt));
    }

    // Initialise derived parameters for variable initialisers
    for(auto &v : m_VarInitialisers) {
        v.second.finalise(dt);
    }
}
//----------------------------------------------------------------------------
bool CurrentSource::isZeroCopyEnabled() const
{
    // If there are any variables implemented in zero-copy mode return true
    return (m_VarLocation.anyZeroCopy() || m_ExtraGlobalParamLocation.anyZeroCopy());
}
//----------------------------------------------------------------------------
bool CurrentSource::isVarInitRequired() const
{
    return std::any_of(m_VarInitialisers.cbegin(), m_VarInitialisers.cend(),
                       [](const auto &init)
                       { 
                           return !Utils::areTokensEmpty(init.second.getCodeTokens());
                       });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CurrentSource::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getCurrentSourceModel()->getHashDigest(), hash);
    Utils::updateHash(getTargetVar(), hash);
    m_DynamicParams.updateHash(hash);

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    for(const auto &v : getNeuronVarReferences()) {
        Utils::updateHash(v.second.getVarName(), hash);
    };
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CurrentSource::getInitHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    Utils::updateHash(getCurrentSourceModel()->getVars(), hash);

    // Include variable initialiser hashes
    for(const auto &w : getVarInitialisers()) {
        Utils::updateHash(w.first, hash);
        Utils::updateHash(w.second.getHashDigest(), hash);
    }
    return hash.get_digest();
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CurrentSource::getVarLocationHashDigest() const
{
    boost::uuids::detail::sha1 hash;
    m_VarLocation.updateHash(hash);
    m_ExtraGlobalParamLocation.updateHash(hash);
    return hash.get_digest();
}
}   // namespace GeNN
