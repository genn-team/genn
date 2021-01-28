#include "varReference.h"

// GeNN includes
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// NeuronVarReference
//----------------------------------------------------------------------------
NeuronVarReference::NeuronVarReference(const NeuronGroup *ng, const std::string &varName) 
: m_NG(static_cast<const NeuronGroupInternal*>(ng))
{
    const auto *nm = ng->getNeuronModel();
    setVar(nm->getVarIndex(varName), nm->getVars());
}
//----------------------------------------------------------------------------
unsigned int NeuronVarReference::getSize() const
{
    return m_NG->getNumNeurons();
}
//----------------------------------------------------------------------------
const NeuronGroup *NeuronVarReference::getNeuronGroup() const 
{ 
    return m_NG; 
}

//----------------------------------------------------------------------------
// CurrentSourceVarReference
//----------------------------------------------------------------------------
CurrentSourceVarReference::CurrentSourceVarReference(const CurrentSource *cs, const std::string &varName)
: m_CS(static_cast<const CurrentSourceInternal*>(cs))
{
    const auto *csm = cs->getCurrentSourceModel();
    setVar(csm->getVarIndex(varName), csm->getVars());
}
//----------------------------------------------------------------------------
unsigned int CurrentSourceVarReference::getSize() const
{
    return m_CS->getTrgNeuronGroup()->getNumNeurons();
}
//----------------------------------------------------------------------------
const CurrentSource *CurrentSourceVarReference::getCurrentSource() const
{
    return m_CS; 
}

//----------------------------------------------------------------------------
// PSMVarReference
//----------------------------------------------------------------------------
PSMVarReference::PSMVarReference(const SynapseGroup *sg, const std::string &varName)
: m_SG(static_cast<const SynapseGroupInternal*>(sg))
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM)) {
        throw std::runtime_error("Only individual postsynaptic model variables can be referenced.");
    }

    const auto *psm = sg->getPSModel();
    setVar(psm->getVarIndex(varName), psm->getVars());
}
//----------------------------------------------------------------------------
unsigned int PSMVarReference::getSize() const
{
    return m_SG->getTrgNeuronGroup()->getNumNeurons();
}
//----------------------------------------------------------------------------
const SynapseGroup *PSMVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}

//----------------------------------------------------------------------------
// WUVarReference
//----------------------------------------------------------------------------
WUVarReference::WUVarReference(const SynapseGroup *sg, const std::string &varName)
: m_SG(static_cast<const SynapseGroupInternal*>(sg))
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
        throw std::runtime_error("Only INDIVIDUAL weight update models can be referenced.");
    }
    const auto *wum = sg->getWUModel();
    setVar(wum->getVarIndex(varName), wum->getVars());
}
//----------------------------------------------------------------------------
unsigned int WUVarReference::getPreSize() const
{
    return m_SG->getSrcNeuronGroup()->getNumNeurons();
}
//----------------------------------------------------------------------------
unsigned int WUVarReference::getMaxRowLength() const
{
    return m_SG->getMaxConnections();
}
//----------------------------------------------------------------------------
const SynapseGroup *WUVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}

//----------------------------------------------------------------------------
// WUPreVarReference
//----------------------------------------------------------------------------
WUPreVarReference::WUPreVarReference(const SynapseGroup *sg, const std::string &varName)
: m_SG(static_cast<const SynapseGroupInternal*>(sg))
{
    const auto *wum = sg->getWUModel();
    setVar(wum->getPreVarIndex(varName), wum->getPreVars());
}
//----------------------------------------------------------------------------
size_t WUPreVarReference::getSize() const
{
    return m_SG->getSrcNeuronGroup()->getNumNeurons();
}
//----------------------------------------------------------------------------
const SynapseGroup *WUPreVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}

//----------------------------------------------------------------------------
// WUPostVarReference
//----------------------------------------------------------------------------
WUPostVarReference::WUPostVarReference(const SynapseGroup *sg, const std::string &varName)
: m_SG(static_cast<const SynapseGroupInternal*>(sg))
{
    const auto *wum = sg->getWUModel();
    setVar(wum->getPostVarIndex(varName), wum->getPostVars());
}
//----------------------------------------------------------------------------
size_t WUPostVarReference::getSize() const
{
    return m_SG->getSrcNeuronGroup()->getNumNeurons();
}
//----------------------------------------------------------------------------
const SynapseGroup *WUPostVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}