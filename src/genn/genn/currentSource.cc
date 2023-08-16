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
    m_VarLocation[getCurrentSourceModel()->getVarIndex(varName)] = loc;
}
//----------------------------------------------------------------------------
void CurrentSource::setExtraGlobalParamLocation(const std::string &paramName, VarLocation loc)
{
    m_ExtraGlobalParamLocation[getCurrentSourceModel()->getExtraGlobalParamIndex(paramName)] = loc;
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
VarLocation CurrentSource::getVarLocation(const std::string &varName) const
{
    return m_VarLocation[getCurrentSourceModel()->getVarIndex(varName)];
}
//----------------------------------------------------------------------------
VarLocation CurrentSource::getExtraGlobalParamLocation(const std::string &varName) const
{
    return m_ExtraGlobalParamLocation[getCurrentSourceModel()->getExtraGlobalParamIndex(varName)];
}
//----------------------------------------------------------------------------
CurrentSource::CurrentSource(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                             const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                             const std::unordered_map<std::string, Models::VarReference> &neuronVarReferences, const NeuronGroupInternal *trgNeuronGroup, 
                             VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
:   m_Name(name), m_CurrentSourceModel(currentSourceModel), m_Params(params), m_VarInitialisers(varInitialisers),
    m_NeuronVarReferences(neuronVarReferences), m_TrgNeuronGroup(trgNeuronGroup), m_VarLocation(varInitialisers.size(), defaultVarLocation),
    m_ExtraGlobalParamLocation(currentSourceModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation),
    m_TargetVar("Isyn")
{
    // Validate names
    Utils::validatePopName(name, "Current source");
    getCurrentSourceModel()->validate(getParams(), getVarInitialisers(), getNeuronVarReferences(), "Current source " + getName());

    // Check variable reference types
    Models::checkVarReferences(getNeuronVarReferences(), getCurrentSourceModel()->getNeuronVarRefs());

    // Check additional local variable reference constraints
    Models::checkLocalVarReferences(getNeuronVarReferences(), getCurrentSourceModel()->getNeuronVarRefs(),
                                    {getTrgNeuronGroup()->getName()}, "Variable references to in current source can only point to target neuron group.");
    
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
    return std::any_of(m_VarLocation.begin(), m_VarLocation.end(),
                       [](VarLocation loc) { return (loc & VarLocation::ZERO_COPY); });
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

    // Loop through neuron variable references and update hash with 
    // name of target variable. These must be the same across merged group
    // as these variable references are just implemented as aliases for neuron variables
    for(const auto &v : getNeuronVarReferences()) {
        Utils::updateHash(v.second.getVar().name, hash);
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
    Utils::updateHash(m_VarLocation, hash);
    Utils::updateHash(m_ExtraGlobalParamLocation, hash);
    return hash.get_digest();
}
}   // namespace GeNN
