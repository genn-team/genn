#pragma once

#include "models.h"
#include "variableMode.h"

// Forward declarations
class CustomConnectivityUpdate;
class CustomUpdateBase;
class NeuronGroup;

//----------------------------------------------------------------------------
// VarAdatorBase
//----------------------------------------------------------------------------
class VarAdaptorBase
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const = 0;

    virtual VarLocation getVarLocation(size_t index) const = 0;
    
    virtual Models::Base::VarVec getVars() const = 0;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const = 0;
};

//----------------------------------------------------------------------------
// NeuronVarAdaptor
//----------------------------------------------------------------------------
class NeuronVarAdaptor : public VarAdaptorBase
{
public:
    NeuronVarAdaptor(const NeuronGroup &ng) : m_NG(ng)
    {}

    //----------------------------------------------------------------------------
    // VarAdaptorBase virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const final;

    virtual VarLocation getVarLocation(size_t index) const final;
    
    virtual Models::Base::VarVec getVars() const final;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const NeuronGroup &m_NG;
};

//----------------------------------------------------------------------------
// SynapseWUVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUVarAdaptor : public VarAdaptorBase
{
public:
    SynapseWUVarAdaptor(const SynapseGroup &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdaptorBase virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const final;

    virtual VarLocation getVarLocation(size_t index) const final;
    
    virtual Models::Base::VarVec getVars() const final;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroup &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUPreVarAdaptor : public VarAdaptorBase
{
public:
    SynapseWUPreVarAdaptor(const SynapseGroup &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdaptorBase virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const final;

    virtual VarLocation getVarLocation(size_t index) const final;
    
    virtual Models::Base::VarVec getVars() const final;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroup &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUPostVarAdaptor : public VarAdaptorBase
{
public:
    SynapseWUPostVarAdaptor(const SynapseGroup &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // VarAdaptorBase virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const final;

    virtual VarLocation getVarLocation(size_t index) const final;
    
    virtual Models::Base::VarVec getVars() const final;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroup &m_SG;
};
//----------------------------------------------------------------------------
// CustomUpdateVarAdaptor
//----------------------------------------------------------------------------
class CustomUpdateVarAdaptor : public VarAdaptorBase
{
public:
    CustomUpdateVarAdaptor(const CustomUpdateBase &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdaptorBase virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const final;

    virtual VarLocation getVarLocation(size_t index) const final;
    
    virtual Models::Base::VarVec getVars() const final;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateBase &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdateVarAdaptor : public VarAdaptorBase
{
public:
    CustomConnectivityUpdateVarAdaptor(const CustomConnectivityUpdate &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdaptorBase virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const final;

    virtual VarLocation getVarLocation(size_t index) const final;
    
    virtual Models::Base::VarVec getVars() const final;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdate &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarAdaptor : public VarAdaptorBase
{
public:
    CustomConnectivityUpdatePreVarAdaptor(const CustomConnectivityUpdate &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdaptorBase virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const final;

    virtual VarLocation getVarLocation(size_t index) const final;
    
    virtual Models::Base::VarVec getVars() const final;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdate &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarAdaptor : public VarAdaptorBase
{
public:
    CustomConnectivityUpdatePostVarAdaptor(const CustomConnectivityUpdate &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // VarAdaptorBase virtuals
    //----------------------------------------------------------------------------
    virtual VarLocation getVarLocation(const std::string &varName) const final;

    virtual VarLocation getVarLocation(size_t index) const final;
    
    virtual Models::Base::VarVec getVars() const final;

    virtual const std::vector<Models::VarInit> &getVarInitialisers() const final;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdate &m_CU;
};

