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
                          const std::unordered_map<std::string, double> &params, const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers,
                          const std::unordered_map<std::string, Models::VarReference> &neuronVarReferences, const NeuronGroupInternal *targetNeuronGroup, 
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
class CurrentSourceVarAdapter
{
public:
    CurrentSourceVarAdapter(const CurrentSourceInternal &cs) : m_CS(cs)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getLoc(const std::string &varName) const{ return m_CS.getVarLocation(varName); }

    std::vector<Models::Base::Var> getDefs() const{ return m_CS.getCurrentSourceModel()->getVars(); }

    const std::unordered_map<std::string, InitVarSnippet::Init> &getInitialisers() const{ return m_CS.getVarInitialisers(); }

    bool isVarDelayed(const std::string&) const{ return false; }

    const CurrentSource &getTarget() const{ return m_CS; }

    VarAccessDim getVarDims(const Models::Base::Var &var) const{ return getVarAccessDim(var.access); }

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

    Snippet::Base::EGPVec getDefs() const{ return m_CS.getCurrentSourceModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};
}   // namespace GeNN
