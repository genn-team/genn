#include "groupEGPAdaptors.h"

// GeNN includes
#include "customUpdateInternal.h"
#include "customConnectivityUpdateInternal.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// NeuronEGPAdaptor
//----------------------------------------------------------------------------
VarLocation NeuronEGPAdaptor::getEGPLocation(const std::string &varName) const
{
    return m_NG.getExtraGlobalParamLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation NeuronEGPAdaptor::getEGPLocation(size_t index) const
{
    return m_NG.getExtraGlobalParamLocation(index);
}
//----------------------------------------------------------------------------    
Snippet::Base::EGPVec NeuronEGPAdaptor::getEGPs() const
{
    return m_NG.getNeuronModel()->getExtraGlobalParams();
}

//----------------------------------------------------------------------------
// CurrentSourceEGPAdaptor
//----------------------------------------------------------------------------
VarLocation CurrentSourceEGPAdaptor::getEGPLocation(const std::string &varName) const
{
    return m_CS.getExtraGlobalParamLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CurrentSourceEGPAdaptor::getEGPLocation(size_t index) const
{
    return m_CS.getExtraGlobalParamLocation(index);
}
//----------------------------------------------------------------------------    
Snippet::Base::EGPVec CurrentSourceEGPAdaptor::getEGPs() const
{
    return m_CS.getCurrentSourceModel()->getExtraGlobalParams();
}

//----------------------------------------------------------------------------
// SynapseWUEGPAdaptor
//----------------------------------------------------------------------------
VarLocation SynapseWUEGPAdaptor::getEGPLocation(const std::string &varName) const
{
    return m_SG.getWUExtraGlobalParamLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapseWUEGPAdaptor::getEGPLocation(size_t index) const
{
    return m_SG.getWUExtraGlobalParamLocation(index);
}
//----------------------------------------------------------------------------
Snippet::Base::EGPVec SynapseWUEGPAdaptor::getEGPs() const
{
    return m_SG.getWUModel()->getExtraGlobalParams();
}


//----------------------------------------------------------------------------
// CustomUpdateEGPAdaptor
//----------------------------------------------------------------------------
VarLocation CustomUpdateEGPAdaptor::getEGPLocation(const std::string &varName) const
{
    // **YUCK**
    return VarLocation::HOST_DEVICE;
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateEGPAdaptor::getEGPLocation(size_t index) const
{
    // **YUCK**
    return VarLocation::HOST_DEVICE;
}
//----------------------------------------------------------------------------    
Snippet::Base::EGPVec CustomUpdateEGPAdaptor::getEGPs() const
{
    return m_CU.getCustomUpdateModel()->getExtraGlobalParams();
}


//----------------------------------------------------------------------------
// CustomConnectivityUpdateEGPAdaptor
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdateEGPAdaptor::getEGPLocation(const std::string &varName) const
{
    // **YUCK**
    return VarLocation::HOST_DEVICE;
}
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdateEGPAdaptor::getEGPLocation(size_t index) const
{
    // **YUCK**
    return VarLocation::HOST_DEVICE;
}
//----------------------------------------------------------------------------
Snippet::Base::EGPVec CustomConnectivityUpdateEGPAdaptor::getEGPs() const
{
    return m_CU.getCustomConnectivityUpdateModel()->getExtraGlobalParams();
}