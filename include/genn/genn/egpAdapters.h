#pragma once

// Standard C++ includes
#include <map>
#include <string>
#include <vector>

// GeNN includes
#include "currentSourceInternal.h"
#include "customConnectivityUpdateInternal.h"
#include "customUpdateInternal.h"
#include "neuronGroupInternal.h"
#include "models.h"
#include "synapseGroupInternal.h"
#include "varLocation.h"

//----------------------------------------------------------------------------
// GeNN::EGPAdapter
//----------------------------------------------------------------------------
namespace GeNN
{
class EGPAdapter
{
public:
    virtual ~EGPAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &name) const = 0;
    
    virtual Snippet::Base::EGPVec getDefs() const = 0;
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


//----------------------------------------------------------------------------
// CustomConnectivityUpdateEGPAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdateEGPAdapter : public EGPAdapter
{
public:
    CustomConnectivityUpdateEGPAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // EGPAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getExtraGlobalParamLocation(varName); }

    virtual Snippet::Base::EGPVec getDefs() const override final { return m_CU.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomUpdateEGPAdapter
//----------------------------------------------------------------------------
class CustomUpdateEGPAdapter : public EGPAdapter
{
public:
    CustomUpdateEGPAdapter(const CustomUpdateBase &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // EGPAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getExtraGlobalParamLocation(varName); }

    virtual Snippet::Base::EGPVec getDefs() const override final { return m_CU.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateBase &m_CU;
};

//----------------------------------------------------------------------------
// NeuronEGPAdapter
//----------------------------------------------------------------------------
class NeuronEGPAdapter : public EGPAdapter
{
public:
    NeuronEGPAdapter(const NeuronGroupInternal &ng) : m_NG(ng)
    {}

    //----------------------------------------------------------------------------
    // EGPAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_NG.getExtraGlobalParamLocation(varName); }

    virtual Snippet::Base::EGPVec getDefs() const override final { return m_NG.getModel()->getExtraGlobalParams(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const NeuronGroupInternal &m_NG;
};

//----------------------------------------------------------------------------
// SynapsePSMEGPAdapter
//----------------------------------------------------------------------------
class SynapsePSMEGPAdapter : public EGPAdapter
{
public:
    SynapsePSMEGPAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // EGPAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getPSExtraGlobalParamLocation(varName); }
    
    virtual Snippet::Base::EGPVec getDefs() const override final { return m_SG.getPSInitialiser().getSnippet()->getExtraGlobalParams(); }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<EGPAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapsePSMEGPAdapter>(sg); }
private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUEGPAdapter
//----------------------------------------------------------------------------
class SynapseWUEGPAdapter : public EGPAdapter
{
public:
    SynapseWUEGPAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // EGPAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getWUExtraGlobalParamLocation(varName); }
    
    virtual Snippet::Base::EGPVec getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getExtraGlobalParams(); }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<EGPAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapseWUEGPAdapter>(sg); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};
}