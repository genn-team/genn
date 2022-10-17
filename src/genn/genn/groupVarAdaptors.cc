#include "groupVarAdaptors.h"

// GeNN includes
#include "customUpdateInternal.h"
#include "customConnectivityUpdateInternal.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// NeuronVarAdaptor
//----------------------------------------------------------------------------
VarLocation NeuronVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_NG.getVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation NeuronVarAdaptor::getVarLocation(size_t index) const
{
    return m_NG.getVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec NeuronVarAdaptor::getVars() const
{
    return m_NG.getNeuronModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &NeuronVarAdaptor::getVarInitialisers() const
{
    return m_NG.getVarInitialisers();
}

//----------------------------------------------------------------------------
// CurrentSourceVarAdaptor
//----------------------------------------------------------------------------
VarLocation CurrentSourceVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_CS.getVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CurrentSourceVarAdaptor::getVarLocation(size_t index) const
{
    return m_CS.getVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CurrentSourceVarAdaptor::getVars() const
{
    return m_CS.getCurrentSourceModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CurrentSourceVarAdaptor::getVarInitialisers() const
{
    return m_CS.getVarInitialisers();
}

//----------------------------------------------------------------------------
// SynapsePSMVarAdaptor
//----------------------------------------------------------------------------
VarLocation SynapsePSMVarAdaptor::getVarLocation(const std::string & varName) const 
{
    return m_SG.getPSVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapsePSMVarAdaptor::getVarLocation(size_t index) const
{
    return m_SG.getPSVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec SynapsePSMVarAdaptor::getVars() const
{
    return m_SG.getPSModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &SynapsePSMVarAdaptor::getVarInitialisers() const
{
    return m_SG.getPSVarInitialisers();
}
//----------------------------------------------------------------------------
const std::string &SynapsePSMVarAdaptor::getFusedVarSuffix() const
{
    return m_SG.getFusedPSVarSuffix();
}

//----------------------------------------------------------------------------
// SynapseWUVarAdaptor
//----------------------------------------------------------------------------
VarLocation SynapseWUVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_SG.getWUVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapseWUVarAdaptor::getVarLocation(size_t index) const
{
    return m_SG.getWUVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec SynapseWUVarAdaptor::getVars() const
{
    return m_SG.getWUModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &SynapseWUVarAdaptor::getVarInitialisers() const
{
    return m_SG.getWUVarInitialisers();
}

//----------------------------------------------------------------------------
// SynapseWUPreVarAdaptor
//----------------------------------------------------------------------------
VarLocation SynapseWUPreVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_SG.getWUPreVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapseWUPreVarAdaptor::getVarLocation(size_t index) const
{
    return m_SG.getWUPreVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec SynapseWUPreVarAdaptor::getVars() const
{
    return m_SG.getWUModel()->getPreVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &SynapseWUPreVarAdaptor::getVarInitialisers() const
{
    return m_SG.getWUPreVarInitialisers();
}
//----------------------------------------------------------------------------
const std::string &SynapseWUPreVarAdaptor::getFusedVarSuffix() const
{
    return m_SG.getFusedWUPreVarSuffix();
}

//----------------------------------------------------------------------------
// SynapseWUPostVarAdaptor
//----------------------------------------------------------------------------
VarLocation SynapseWUPostVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_SG.getWUPostVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation SynapseWUPostVarAdaptor::getVarLocation(size_t index) const
{
    return m_SG.getWUPostVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec SynapseWUPostVarAdaptor::getVars() const
{
    return m_SG.getWUModel()->getPostVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &SynapseWUPostVarAdaptor::getVarInitialisers() const
{
    return m_SG.getWUPostVarInitialisers();
}
//----------------------------------------------------------------------------
const std::string &SynapseWUPostVarAdaptor::getFusedVarSuffix() const
{
    return m_SG.getFusedWUPostVarSuffix();
}

//----------------------------------------------------------------------------
// CustomUpdateVarAdaptor
//----------------------------------------------------------------------------
VarLocation CustomUpdateVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_CU.getVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CustomUpdateVarAdaptor::getVarLocation(size_t index) const
{
    return m_CU.getVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CustomUpdateVarAdaptor::getVars() const
{
    return m_CU.getCustomUpdateModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CustomUpdateVarAdaptor::getVarInitialisers() const
{
    return m_CU.getVarInitialisers();
}

//----------------------------------------------------------------------------
// CustomConnectivityUpdateVarAdaptor
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdateVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_CU.getVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdateVarAdaptor::getVarLocation(size_t index) const
{
    return m_CU.getVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CustomConnectivityUpdateVarAdaptor::getVars() const
{
    return m_CU.getCustomConnectivityUpdateModel()->getVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CustomConnectivityUpdateVarAdaptor::getVarInitialisers() const
{
    return m_CU.getVarInitialisers();
}

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePreVarAdaptor
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdatePreVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_CU.getPreVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdatePreVarAdaptor::getVarLocation(size_t index) const
{
    return m_CU.getPreVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CustomConnectivityUpdatePreVarAdaptor::getVars() const
{
    return m_CU.getCustomConnectivityUpdateModel()->getPreVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CustomConnectivityUpdatePreVarAdaptor::getVarInitialisers() const
{
    return m_CU.getPreVarInitialisers();
}

//----------------------------------------------------------------------------
// CustomConnectivityUpdatePostVarAdaptor
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdatePostVarAdaptor::getVarLocation(const std::string &varName) const
{
    return m_CU.getPostVarLocation(varName);
}
//----------------------------------------------------------------------------
VarLocation CustomConnectivityUpdatePostVarAdaptor::getVarLocation(size_t index) const
{
    return m_CU.getPostVarLocation(index);
}
//----------------------------------------------------------------------------
Models::Base::VarVec CustomConnectivityUpdatePostVarAdaptor::getVars() const
{
    return m_CU.getCustomConnectivityUpdateModel()->getPostVars();
}
//----------------------------------------------------------------------------
const std::vector<Models::VarInit> &CustomConnectivityUpdatePostVarAdaptor::getVarInitialisers() const
{
    return m_CU.getPostVarInitialisers();
}
