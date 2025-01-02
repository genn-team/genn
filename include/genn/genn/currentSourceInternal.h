#pragma once

// GeNN includes
#include "adapters.h"
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
                          const std::map<std::string, Models::VarReference> &neuronVarReferences, const NeuronGroupInternal *targetNeuronGroup, 
                          VarLocation defaultVarLocation, VarLocation defaultExtraGlobalParamLocation)
    :   CurrentSource(name, currentSourceModel, params, varInitialisers, neuronVarReferences, 
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
class CurrentSourceVarAdapter : public VarAdapter
{
public:
    CurrentSourceVarAdapter(const CurrentSourceInternal &cs) : m_CS(cs)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CS.getVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_CS.getModel()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_CS.getVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final { return std::nullopt; }

    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    const auto &getTarget() const{ return m_CS; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};

//----------------------------------------------------------------------------
// CurrentSourceNeuronVarRefAdapter
//----------------------------------------------------------------------------
class CurrentSourceNeuronVarRefAdapter : public VarRefAdapter
{
public:
    CurrentSourceNeuronVarRefAdapter(const CurrentSourceInternal &cs) : m_CS(cs)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const final override{ return m_CS.getModel()->getNeuronVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const final override { return m_CS.getNeuronVarReferences(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};

//----------------------------------------------------------------------------
// CurrentSourceEGPAdapter
//----------------------------------------------------------------------------
class CurrentSourceEGPAdapter : public EGPAdapter
{
public:
    CurrentSourceEGPAdapter(const CurrentSourceInternal &cs) : m_CS(cs)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const final override { return m_CS.getExtraGlobalParamLocation(varName); }

    virtual Snippet::Base::EGPVec getDefs() const final override { return m_CS.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};
}   // namespace GeNN
