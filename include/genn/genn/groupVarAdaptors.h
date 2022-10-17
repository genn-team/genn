#pragma once

// GeNN includes
#include "models.h"
#include "variableMode.h"

// Forward declarations
class CurrentSourceInternal;
class CustomConnectivityUpdateInternal;
class CustomUpdateBase;
class NeuronGroupInternal;
class SynapseGroupInternal;

//----------------------------------------------------------------------------
// NeuronVarAdaptor
//----------------------------------------------------------------------------
class NeuronVarAdaptor
{
public:
    NeuronVarAdaptor(const NeuronGroupInternal &ng) : m_NG(ng)
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
    const NeuronGroupInternal &m_NG;
};

//----------------------------------------------------------------------------
// CurrentSourceVarAdaptor
//----------------------------------------------------------------------------
class CurrentSourceVarAdaptor
{
public:
    CurrentSourceVarAdaptor(const CurrentSourceInternal &cs) : m_CS(cs)
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
    const CurrentSourceInternal &m_CS;
};

//----------------------------------------------------------------------------
// SynapsePSMVarAdaptor
//----------------------------------------------------------------------------
class SynapsePSMVarAdaptor
{
public:
    SynapsePSMVarAdaptor(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

    const std::string &getFusedVarSuffix() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUVarAdaptor
{
public:
    SynapseWUVarAdaptor(const SynapseGroupInternal &sg) : m_SG(sg)
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
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPreVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUPreVarAdaptor
{
public:
    SynapseWUPreVarAdaptor(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

    const std::string &getFusedVarSuffix() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// SynapseWUPostVarAdaptor
//----------------------------------------------------------------------------
class SynapseWUPostVarAdaptor
{
public:
    SynapseWUPostVarAdaptor(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getVarLocation(const std::string &varName) const;

    VarLocation getVarLocation(size_t index) const;
    
    Models::Base::VarVec getVars() const;

    const std::vector<Models::VarInit> &getVarInitialisers() const;

    const std::string &getFusedVarSuffix() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
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
    CustomConnectivityUpdateVarAdaptor(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
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
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePreVarAdaptor
{
public:
    CustomConnectivityUpdatePreVarAdaptor(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
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
    const CustomConnectivityUpdateInternal &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdatePostVarAdaptor
{
public:
    CustomConnectivityUpdatePostVarAdaptor(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
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
    const CustomConnectivityUpdateInternal &m_CU;
};

