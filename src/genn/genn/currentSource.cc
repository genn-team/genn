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
    m_ExtraGlobalParamLocation(currentSourceModel->getExtraGlobalParams().size(), defaultExtraGlobalParamLocation)
{
    // Validate names
    Utils::validatePopName(name, "Current source");
    getCurrentSourceModel()->validate(getParams(), getVarInitialisers(), getNeuronVarReferences(), "Current source " + getName());

    // Check variable reference types
    Models::checkVarReferences(m_NeuronVarReferences, getCurrentSourceModel()->getNeuronVarRefs());

    // Loop through all variable references
    for(const auto &modelVarRef : getCurrentSourceModel()->getNeuronVarRefs()) {
        const auto &varRef = m_NeuronVarReferences.at(modelVarRef.name);

        // If neuron var reference point to any SHARED_NEURON or SHARED variables, check that access is read-only
        if(((varRef.getVar().access & VarAccessDuplication::SHARED_NEURON) || (varRef.getVar().access & VarAccessDuplication::SHARED))
            && (modelVarRef.access != VarAccessMode::READ_ONLY))
        {
            throw std::runtime_error("Variable references to SHARED_NEURON or SHARED neuron variables in current source cannot be read-write.");
        }

        // Check variable reference points to target neuron
        // **YUCK** this check works but is a bit gross
        if(varRef.getTargetName() != getTrgNeuronGroup()->getName()) {
            throw std::runtime_error("Variable references to in current source can only point to target neuron group.");
        }
    }

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
        
    // Loop through neuron variable references and update hash with 
    // duplication mode of target variable as this effects indexing code
    for(const auto &v : getNeuronVarReferences()) {
        Utils::updateHash(getVarAccessDuplication(v.second.getVar().access), hash);
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
