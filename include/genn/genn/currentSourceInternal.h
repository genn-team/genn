#pragma once

// GeNN includes
#include "currentSource.h"

//------------------------------------------------------------------------
// GeNN::CurrentSourceInternal
//------------------------------------------------------------------------
namespace GeNN
{
class CurrentSourceInternal : public CurrentSource
{
public:
    using GroupExternal = CurrentSource;

    CurrentSourceInternal(const std::string &name, const CurrentSourceModels::Base *currentSourceModel,
                          const std::map<std::string, Type::NumericValue> &params, const std::map<std::string, InitVarSnippet::Init> &varInitialisers,
                          const std::map<std::string, std::variant<std::string, Models::VarReference>> &neuronVarReferences,
                          const std::map<std::string, std::variant<std::string, Models::EGPReference>> &neuronEGPReferences,
                          NeuronGroupInternal *targetNeuronGroup, VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   CurrentSource(name, currentSourceModel, params, varInitialisers, neuronVarReferences, neuronEGPReferences,
                      targetNeuronGroup, defaultVarLocation, defaultExtraGlobalParamLocation)
    {
    }

    using CurrentSource::getTrgNeuronGroup;
    using CurrentSource::finalise;
    using CurrentSource::getDerivedParams;
    using CurrentSource::isZeroCopyEnabled;
    using CurrentSource::isVarInitRequired;
    using CurrentSource::getHashDigest;
    using CurrentSource::getInitHashDigest;
    using CurrentSource::getVarLocationHashDigest;
    using CurrentSource::getInjectionCodeTokens;
};

//----------------------------------------------------------------------------
// CurrentSourceVarAdapter
//----------------------------------------------------------------------------
class CurrentSourceVarAdapter
{
public:
    CurrentSourceVarAdapter(const CurrentSourceInternal &cs) : m_CS(cs)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CS.getVarLocation(varName); }

    auto getDefs() const{ return m_CS.getModel()->getVars(); }

    const auto &getInitialisers() const{ return m_CS.getVarInitialisers(); }

    std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const{ return std::nullopt; }

    const auto &getTarget() const{ return m_CS; }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};

//----------------------------------------------------------------------------
// CurrentSourceNeuronVarRefAdapter
//----------------------------------------------------------------------------
class CurrentSourceNeuronVarRefAdapter
{
public:
    CurrentSourceNeuronVarRefAdapter(const CurrentSourceInternal &cs) : m_CS(cs)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    auto getDefs() const{ return m_CS.getModel()->getNeuronVarRefs(); }

    const auto &getInitialisers() const{ return m_CS.getNeuronVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};

//----------------------------------------------------------------------------
// CurrentSourceEGPAdapter
//----------------------------------------------------------------------------
class CurrentSourceEGPAdapter
{
public:
    CurrentSourceEGPAdapter(const CurrentSourceInternal &cs) : m_CS(cs)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CS.getExtraGlobalParamLocation(varName); }

    auto getDefs() const{ return m_CS.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};
}   // namespace GeNN
