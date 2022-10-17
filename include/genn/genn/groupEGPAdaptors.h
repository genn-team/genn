#pragma once

// GeNN includes

#include "models.h"
#include "variableMode.h"

// Forward declarations
class CurrentSourceInternal;
class CustomConnectivityUpdateInternal;
class CustomUpdateBase;
class NeuronGroupInternal;

//----------------------------------------------------------------------------
// NeuronEGPAdaptor
//----------------------------------------------------------------------------
class NeuronEGPAdaptor
{
public:
    NeuronEGPAdaptor(const NeuronGroupInternal &ng) : m_NG(ng)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getEGPLocation(const std::string &varName) const;

    VarLocation getEGPLocation(size_t index) const;
    
    Snippet::Base::EGPVec getEGPs() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const NeuronGroupInternal &m_NG;
};


//----------------------------------------------------------------------------
// CurrentSourceEGPAdaptor
//----------------------------------------------------------------------------
class CurrentSourceEGPAdaptor
{
public:
    CurrentSourceEGPAdaptor(const CurrentSourceInternal &cs) : m_CS(cs)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getEGPLocation(const std::string &varName) const;

    VarLocation getEGPLocation(size_t index) const;
    
    Snippet::Base::EGPVec getEGPs() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CurrentSourceInternal &m_CS;
};

//----------------------------------------------------------------------------
// SynapseWUEGPAdaptor
//----------------------------------------------------------------------------
class SynapseWUEGPAdaptor
{
public:
    SynapseWUEGPAdaptor(const SynapseGroupInternal &sg) : m_SG(sg)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getEGPLocation(const std::string &varName) const;

    VarLocation getEGPLocation(size_t index) const;
    
    Snippet::Base::EGPVec getEGPs() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const SynapseGroupInternal &m_SG;
};

//----------------------------------------------------------------------------
// CustomUpdateEGPAdaptor
//----------------------------------------------------------------------------
class CustomUpdateEGPAdaptor
{
public:
    CustomUpdateEGPAdaptor(const CustomUpdateBase &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getEGPLocation(const std::string &varName) const;

    VarLocation getEGPLocation(size_t index) const;
    
    Snippet::Base::EGPVec getEGPs() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomUpdateBase &m_CU;
};

//----------------------------------------------------------------------------
// CustomConnectivityUpdateEGPAdaptor
//----------------------------------------------------------------------------
class CustomConnectivityUpdateEGPAdaptor
{
public:
    CustomConnectivityUpdateEGPAdaptor(const CustomConnectivityUpdateInternal &cu) : m_CU(cu)
    {}

    //----------------------------------------------------------------------------
    // Public methods
    //----------------------------------------------------------------------------
    VarLocation getEGPLocation(const std::string &varName) const;

    VarLocation getEGPLocation(size_t index) const;
    
    Snippet::Base::EGPVec getEGPs() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const CustomConnectivityUpdateInternal &m_CU;
};