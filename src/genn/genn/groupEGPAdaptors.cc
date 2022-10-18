#include "groupEGPAdaptors.h"

// GeNN includes
#include "customUpdateInternal.h"
#include "customConnectivityUpdateInternal.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// NeuronEGPAdapter
//----------------------------------------------------------------------------
VarLocation NeuronEGPAdapter::getEGPLocation(const std::string &varName) const
{
    return m_NG.getExtraGlobalParamLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation NeuronEGPAdapter::getEGPLocation(size_t index) const
{
    return m_NG.getExtraGlobalParamLocation(index);
}
//----------------------------------------------------------------------------    
Snippet::Base::EGPVec NeuronEGPAdapter::getEGPs() const
{
    return m_NG.getNeuronModel()->getExtraGlobalParams();
}

//----------------------------------------------------------------------------
// CurrentSourceEGPAdapter
//----------------------------------------------------------------------------
VarLocation CurrentSourceEGPAdapter::getEGPLocation(const std::string &varName) const
{
    return m_CS.getExtraGlobalParamLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CurrentSourceEGPAdapter::getEGPLocation(size_t index) const
{
    return m_CS.getExtraGlobalParamLocation(index);
}
//----------------------------------------------------------------------------    
Snippet::Base::EGPVec CurrentSourceEGPAdapter::getEGPs() const
{
    return m_CS.getCurrentSourceModel()->getExtraGlobalParams();
}

//----------------------------------------------------------------------------
// SynapseWUEGPAdapter
//----------------------------------------------------------------------------
VarLocation SynapseWUEGPAdapter::getEGPLocation(const std::string &varName) const
{
    return m_SG.getWUExtraGlobalParamLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapseWUEGPAdapter::getEGPLocation(size_t index) const
{
    return m_SG.getWUExtraGlobalParamLocation(index);
}
//----------------------------------------------------------------------------
Snippet::Base::EGPVec SynapseWUEGPAdapter::getEGPs() const
{
    return m_SG.getWUModel()->getExtraGlobalParams();
}


//----------------------------------------------------------------------------
// CustomUpdateEGPAdapter
//----------------------------------------------------------------------------
VarLocation CustomUpdateEGPAdapter::getEGPLocation(const std::string &varName) const
{
    // **YUCK**
    return VarLocation::HOST_DEVICE;
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateEGPAdapter::getEGPLocation(size_t index) const
{
    // **YUCK**
    return VarLocation::HOST_DEVICE;
}
//----------------------------------------------------------------------------    
Snippet::Base::EGPVec CustomUpdateEGPAdapter::getEGPs() const
{
    return m_CU.getCustomUpdateModel()->getExtraGlobalParams();
}


//----------------------------------------------------------------------------
// CustomConnectivityUpdateEGPAdapter
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdateEGPAdapter::getEGPLocation(const std::string &varName) const
{
    // **YUCK**
    return VarLocation::HOST_DEVICE;
}
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdateEGPAdapter::getEGPLocation(size_t index) const
{
    // **YUCK**
    return VarLocation::HOST_DEVICE;
}
//----------------------------------------------------------------------------
Snippet::Base::EGPVec CustomConnectivityUpdateEGPAdapter::getEGPs() const
{
    return m_CU.getCustomConnectivityUpdateModel()->getExtraGlobalParams();
}