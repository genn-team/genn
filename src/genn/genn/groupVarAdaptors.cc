#include "groupVarAdaptors.h"

// GeNN includes
#include "customUpdateInternal.h"
#include "customConnectivityUpdateInternal.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// NeuronVarAdapter
//----------------------------------------------------------------------------
VarLocation NeuronVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_NG.getVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation NeuronVarAdapter::getVarLocation(size_t index) const
{
    return m_NG.getVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec NeuronVarAdapter::getVars() const
{
    return m_NG.getNeuronModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &NeuronVarAdapter::getVarInitialisers() const
{
    return m_NG.getVarInitialisers();
}

//----------------------------------------------------------------------------
// CurrentSourceVarAdapter
//----------------------------------------------------------------------------
VarLocation CurrentSourceVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_CS.getVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CurrentSourceVarAdapter::getVarLocation(size_t index) const
{
    return m_CS.getVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CurrentSourceVarAdapter::getVars() const
{
    return m_CS.getCurrentSourceModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CurrentSourceVarAdapter::getVarInitialisers() const
{
    return m_CS.getVarInitialisers();
}

//----------------------------------------------------------------------------
// SynapsePSMVarAdapter
//----------------------------------------------------------------------------
VarLocation SynapsePSMVarAdapter::getVarLocation(const std::string & varName) const 
{
    return m_SG.getPSVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapsePSMVarAdapter::getVarLocation(size_t index) const
{
    return m_SG.getPSVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec SynapsePSMVarAdapter::getVars() const
{
    return m_SG.getPSModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &SynapsePSMVarAdapter::getVarInitialisers() const
{
    return m_SG.getPSVarInitialisers();
}
//----------------------------------------------------------------------------
const std::string &SynapsePSMVarAdapter::getFusedVarSuffix() const
{
    return m_SG.getFusedPSVarSuffix();
}

//----------------------------------------------------------------------------
// SynapseWUVarAdapter
//----------------------------------------------------------------------------
VarLocation SynapseWUVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_SG.getWUVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapseWUVarAdapter::getVarLocation(size_t index) const
{
    return m_SG.getWUVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec SynapseWUVarAdapter::getVars() const
{
    return m_SG.getWUModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &SynapseWUVarAdapter::getVarInitialisers() const
{
    return m_SG.getWUVarInitialisers();
}

//----------------------------------------------------------------------------
// SynapseWUPreVarAdapter
//----------------------------------------------------------------------------
VarLocation SynapseWUPreVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_SG.getWUPreVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapseWUPreVarAdapter::getVarLocation(size_t index) const
{
    return m_SG.getWUPreVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec SynapseWUPreVarAdapter::getVars() const
{
    return m_SG.getWUModel()->getPreVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &SynapseWUPreVarAdapter::getVarInitialisers() const
{
    return m_SG.getWUPreVarInitialisers();
}
//----------------------------------------------------------------------------
const std::string &SynapseWUPreVarAdapter::getFusedVarSuffix() const
{
    return m_SG.getFusedWUPreVarSuffix();
}

//----------------------------------------------------------------------------
// SynapseWUPostVarAdapter
//----------------------------------------------------------------------------
VarLocation SynapseWUPostVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_SG.getWUPostVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapseWUPostVarAdapter::getVarLocation(size_t index) const
{
    return m_SG.getWUPostVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec SynapseWUPostVarAdapter::getVars() const
{
    return m_SG.getWUModel()->getPostVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &SynapseWUPostVarAdapter::getVarInitialisers() const
{
    return m_SG.getWUPostVarInitialisers();
}
//----------------------------------------------------------------------------
const std::string &SynapseWUPostVarAdapter::getFusedVarSuffix() const
{
    return m_SG.getFusedWUPostVarSuffix();
}

//----------------------------------------------------------------------------
// CustomUpdateVarAdapter
//----------------------------------------------------------------------------
VarLocation CustomUpdateVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_CU.getVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateVarAdapter::getVarLocation(size_t index) const
{
    return m_CU.getVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CustomUpdateVarAdapter::getVars() const
{
    return m_CU.getCustomUpdateModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CustomUpdateVarAdapter::getVarInitialisers() const
{
    return m_CU.getVarInitialisers();
}

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarAdapter
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdateVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_CU.getVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdateVarAdapter::getVarLocation(size_t index) const
{
    return m_CU.getVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CustomConnectivityUpdateVarAdapter::getVars() const
{
    return m_CU.getCustomConnectivityUpdateModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CustomConnectivityUpdateVarAdapter::getVarInitialisers() const
{
    return m_CU.getVarInitialisers();
}

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdapter
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdatePreVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_CU.getPreVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdatePreVarAdapter::getVarLocation(size_t index) const
{
    return m_CU.getPreVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CustomConnectivityUpdatePreVarAdapter::getVars() const
{
    return m_CU.getCustomConnectivityUpdateModel()->getPreVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CustomConnectivityUpdatePreVarAdapter::getVarInitialisers() const
{
    return m_CU.getPreVarInitialisers();
}

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarAdapter
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdatePostVarAdapter::getVarLocation(const std::string &varName) const
{
    return m_CU.getPostVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdatePostVarAdapter::getVarLocation(size_t index) const
{
    return m_CU.getPostVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CustomConnectivityUpdatePostVarAdapter::getVars() const
{
    return m_CU.getCustomConnectivityUpdateModel()->getPostVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CustomConnectivityUpdatePostVarAdapter::getVarInitialisers() const
{
    return m_CU.getPostVarInitialisers();
}
