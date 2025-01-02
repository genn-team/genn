#pragma once

// Standard C++ includes
#include <map>
#include <optional>
#include <string>
#include <vector>

// GeNN includes
#include "currentSourceInternal.h"
#include "customConnectivityUpdateInternal.h"
#include "customUpdateInternal.h"
#include "models.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"
#include "varLocation.h"

// Forward declarations
namespace GeNN::Runtime
{
class ArrayBase;
class Runtime;
}

//----------------------------------------------------------------------------
// GeNN::VarAdapterBase
//----------------------------------------------------------------------------
namespace GeNN
{
class VarAdapterBase
{
public:
    virtual ~VarAdapterBase() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &name) const = 0;
   
    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const = 0;

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const = 0;

    //! Get array associated with variable
    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const = 0;
};

//----------------------------------------------------------------------------
// GeNN::VarAdapter
//----------------------------------------------------------------------------
class VarAdapter : public VarAdapterBase
{
public:
    virtual ~VarAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Models::Base::Var> getDefs() const = 0;
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const = 0;
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

    //! Get array associated with variable
    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdateVarAdapter : public VarAdapter
{
public:
    CustomConnectivityUpdateVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_CU.getModel()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_CU.getVarInitialisers(); }
    
    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final { return std::nullopt; }

    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //! Get array associated with variable
    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarAdapter : public VarAdapter
{
public:
    CustomConnectivityUpdatePreVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getPreVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_CU.getModel()->getPreVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_CU.getPreVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final { return std::nullopt; }

    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    //! Get array associated with variable
    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarAdapter : public VarAdapter
{
public:
    CustomConnectivityUpdatePostVarAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getPostVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_CU.getModel()->getPostVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_CU.getPostVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final { return std::nullopt; }
    
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// NeuronVarAdapter
//----------------------------------------------------------------------------
class NeuronVarAdapter : public VarAdapter
{
public:
    NeuronVarAdapter(const NeuronGroupInternal &ng) : m_NG(ng)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_NG.getVarLocation(varName); }
    
    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_NG.getModel()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_NG.getVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final
    { 
        if(m_NG.isDelayRequired() && m_NG.isVarQueueRequired(varName)) {
            return m_NG.getNumDelaySlots();
        }
        else {
            return std::nullopt; 
        }
    }

    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const NeuronGroupInternal &m_NG;
};

//----------------------------------------------------------------------------
// SynapsePSMVarAdapter
//----------------------------------------------------------------------------
class SynapsePSMVarAdapter : public VarAdapter
{
public:
    SynapsePSMVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getPSVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_SG.getPSInitialiser().getSnippet()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_SG.getPSInitialiser().getVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final { return std::nullopt; }
    
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapsePSMVarAdapter>(sg); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUVarAdapter
//----------------------------------------------------------------------------
class SynapseWUVarAdapter : public VarAdapter
{
public:
    SynapseWUVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getWUVarLocation(varName); }
    
    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_SG.getWUInitialiser().getVarInitialisers(); }
    
    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const{ return std::nullopt; }

    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapseWUVarAdapter>(sg); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreVarAdapter
//----------------------------------------------------------------------------
class SynapseWUPreVarAdapter : public VarAdapter
{
public:
    SynapseWUPreVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getWUPreVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getPreVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_SG.getWUInitialiser().getPreVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final
    {
        if(m_SG.getAxonalDelaySteps() != 0) {
            return m_SG.getSrcNeuronGroup()->getNumDelaySlots();
        }
        else {
            return std::nullopt;
        }
    }
    
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }

    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapseWUPreVarAdapter>(sg); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostVarAdapter
//----------------------------------------------------------------------------
class SynapseWUPostVarAdapter : public VarAdapter
{
public:
    SynapseWUPostVarAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_SG.getWUPostVarLocation(varName); }

    virtual std::vector<Models::Base::Var> getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getPostVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_SG.getWUInitialiser().getPostVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final
    {
        if(m_SG.getBackPropDelaySteps() != 0 || m_SG.isWUPostVarHeterogeneouslyDelayed(varName)) {
            return m_SG.getTrgNeuronGroup()->getNumDelaySlots();
        }
        else {
            return std::nullopt;
        }
    }
    
    virtual VarAccessDim getVarDims(const Models::Base::Var &var) const override final { return getVarAccessDim(var.access); }
    
    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapseWUPostVarAdapter>(sg); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// GeNN::CUVarAdapter
//----------------------------------------------------------------------------
class CUVarAdapter : public VarAdapterBase
{
public:
    virtual ~CUVarAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual std::vector<Models::Base::CustomUpdateVar> getDefs() const = 0;
    virtual VarAccessDim getVarDims(const Models::Base::CustomUpdateVar &var) const = 0;
};

//----------------------------------------------------------------------------
// CustomUpdateVarAdapter
//----------------------------------------------------------------------------
class CustomUpdateVarAdapter : public CUVarAdapter
{
public:
    CustomUpdateVarAdapter(const CustomUpdateBase &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdapter virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getLoc(const std::string &varName) const override final { return m_CU.getVarLocation(varName); }

    virtual std::vector<Models::Base::CustomUpdateVar> getDefs() const override final { return m_CU.getModel()->getVars(); }

    virtual const std::map<std::string, InitVarSnippet::Init> &getInitialisers() const override final { return m_CU.getVarInitialisers(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string&) const override final { return std::nullopt; }
    
    virtual VarAccessDim getVarDims(const Models::Base::CustomUpdateVar &var) const override final
    { 
        return getVarAccessDim(var.access, m_CU.getDims());
    }

    virtual const Runtime::ArrayBase *getTargetArray(const Runtime::Runtime &runtime, const std::string &varName) const override final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateBase &m_CU;
};
}