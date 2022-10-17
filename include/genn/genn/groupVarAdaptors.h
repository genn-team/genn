#pragma once

#include "models.h"
#include "variableMode.h"

// Forward declarations
class CustomConnectivityUpdate;
class CustomUpdateBase;
class NeuronGroup;

//----------------------------------------------------------------------------
// NeuronVarAdaptor
//----------------------------------------------------------------------------
class NeuronVarAdaptor
{
public:
    NeuronVarAdaptor(const NeuronGroup &ng) : m_NG(ng)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const NeuronGroup &m_NG;
};

//----------------------------------------------------------------------------
// SynapseWUVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUVarAdaptor
{
public:
    SynapseWUVarAdaptor(const SynapseGroup &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroup &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUPreVarAdaptor
{
public:
    SynapseWUPreVarAdaptor(const SynapseGroup &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroup &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUPostVarAdaptor
{
public:
    SynapseWUPostVarAdaptor(const SynapseGroup &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroup &m_SG;
};
//----------------------------------------------------------------------------
// CustomUpdateVarAdaptor
//----------------------------------------------------------------------------
class CustomUpdateVarAdaptor
{
public:
    CustomUpdateVarAdaptor(const CustomUpdateBase &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateBase &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdateVarAdaptor
{
public:
    CustomConnectivityUpdateVarAdaptor(const CustomConnectivityUpdate &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdate &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarAdaptor
{
public:
    CustomConnectivityUpdatePreVarAdaptor(const CustomConnectivityUpdate &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdate &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarAdaptor
{
public:
    CustomConnectivityUpdatePostVarAdaptor(const CustomConnectivityUpdate &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdate &m_CU;
};

