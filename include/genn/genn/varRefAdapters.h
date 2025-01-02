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
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// GeNN::VarRefAdapterBase
//----------------------------------------------------------------------------
namespace GeNN
{
class VarRefAdapterBase
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals 
    //------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const = 0;

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varRefName) const = 0;
};

//----------------------------------------------------------------------------
// GeNN::VarRefAdapter
//----------------------------------------------------------------------------
class VarRefAdapter : public VarRefAdapterBase
{
public:
    virtual ~VarRefAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const = 0;
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

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarRefAdapter> create(const CurrentSourceInternal &cs){ return std::make_unique<CurrentSourceNeuronVarRefAdapter>(cs); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarRefAdapter : public VarRefAdapter
{
public:
    CustomConnectivityUpdatePreVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_CU.getModel()->getPreVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_CU.getPreVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarRefAdapter> create(const CustomConnectivityUpdateInternal &cu){ return std::make_unique<CustomConnectivityUpdatePreVarRefAdapter>(cu); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarRefAdapter : public VarRefAdapter
{
public:
    CustomConnectivityUpdatePostVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_CU.getModel()->getPostVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_CU.getPostVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarRefAdapter> create(const CustomConnectivityUpdateInternal &cu){ return std::make_unique<CustomConnectivityUpdatePostVarRefAdapter>(cu); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomUpdateVarRefAdapter
//----------------------------------------------------------------------------
class CustomUpdateVarRefAdapter : public VarRefAdapter
{
public:
    CustomUpdateVarRefAdapter(const CustomUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    Models::Base::VarRefVec getDefs() const override final { return m_CU.getModel()->getVarRefs(); }

    const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_CU.getVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final
    {
        const auto &varRef = m_CU.getVarReferences().at(varName);
        const auto *delayNeuronGroup = varRef.getDelayNeuronGroup();
        const auto *denDelaySynapseGroup = varRef.getDenDelaySynapseGroup();
        if(delayNeuronGroup) {
            return delayNeuronGroup->getNumDelaySlots();
        }
        else if(denDelaySynapseGroup) {
            return denDelaySynapseGroup->getMaxDendriticDelayTimesteps();
        }
        else {
            return std::nullopt;
        }
    }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarRefAdapter> create(const CustomUpdateInternal &cu){ return std::make_unique<CustomUpdateVarRefAdapter>(cu); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateInternal &m_CU;
};


//----------------------------------------------------------------------------
// SynapsePSMNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapsePSMNeuronVarRefAdapter : public VarRefAdapter
{
public:
    SynapsePSMNeuronVarRefAdapter(const SynapseGroupInternal &cs) : m_SG(cs)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_SG.getPSInitialiser().getSnippet()->getNeuronVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_SG.getPSInitialiser().getNeuronVarReferences(); }
    
    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarRefAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapsePSMNeuronVarRefAdapter>(sg); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapseWUPreNeuronVarRefAdapter : public VarRefAdapter
{
public:
    SynapseWUPreNeuronVarRefAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getPreNeuronVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_SG.getWUInitialiser().getPreNeuronVarReferences(); }
    
    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarRefAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapseWUPreNeuronVarRefAdapter>(sg); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostNeuronVarRefAdapter
//----------------------------------------------------------------------------
class SynapseWUPostNeuronVarRefAdapter : public VarRefAdapter
{
public:
    SynapseWUPostNeuronVarRefAdapter(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    using RefType = Models::VarReference;

    //----------------------------------------------------------------------------
    // VarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_SG.getWUInitialiser().getSnippet()->getPostNeuronVarRefs(); }

    virtual const std::map<std::string, Models::VarReference> &getInitialisers() const override final { return m_SG.getWUInitialiser().getPostNeuronVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final{ throw std::runtime_error("Not implemented"); }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<VarRefAdapter> create(const SynapseGroupInternal &sg){ return std::make_unique<SynapseWUPostNeuronVarRefAdapter>(sg); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// GeNN::WUVarRefAdapter
//----------------------------------------------------------------------------
class WUVarRefAdapter : public VarRefAdapterBase
{
public:
    virtual ~WUVarRefAdapter() = default;

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual const std::map<std::string, Models::WUVarReference> &getInitialisers() const = 0;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarRefAdapter
//----------------------------------------------------------------------------
class CustomConnectivityUpdateVarRefAdapter : public WUVarRefAdapter
{
public:
    CustomConnectivityUpdateVarRefAdapter(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::WUVarReference;

    //----------------------------------------------------------------------------
    // WUVarRefAdapter virtuals
    //----------------------------------------------------------------------------
    virtual Models::Base::VarRefVec getDefs() const override final { return m_CU.getModel()->getVarRefs(); }

    virtual const std::map<std::string, Models::WUVarReference> &getInitialisers() const override final { return m_CU.getVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final { return std::nullopt; }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<WUVarRefAdapter> create(const CustomConnectivityUpdateInternal &cu){ return std::make_unique<CustomConnectivityUpdateVarRefAdapter>(cu); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomUpdateWUVarRefAdapter
//----------------------------------------------------------------------------
class CustomUpdateWUVarRefAdapter : public WUVarRefAdapter
{
public:
    CustomUpdateWUVarRefAdapter(const CustomUpdateWUInternal &cu) : m_CU(cu)
    {}

    using RefType = Models::WUVarReference;

    //----------------------------------------------------------------------------
    // WUVarRefAdapter virtuals
    //----------------------------------------------------------------------------
    Models::Base::VarRefVec getDefs() const override final { return m_CU.getModel()->getVarRefs(); }

    const std::map<std::string, Models::WUVarReference> &getInitialisers() const override final { return m_CU.getVarReferences(); }

    virtual std::optional<unsigned int> getNumVarDelaySlots(const std::string &varName) const override final { return std::nullopt; }

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::unique_ptr<WUVarRefAdapter> create(const CustomUpdateWUInternal &cu){ return std::make_unique<CustomUpdateWUVarRefAdapter>(cu); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateWUInternal &m_CU;
};
}