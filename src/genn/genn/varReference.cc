#include "varReference.h"

// GeNN includes
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// NeuronVarReference
//----------------------------------------------------------------------------
const std::string NeuronVarReference::name = "Neuron";
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
const std::string &NeuronVarReference::getTargetName() const
{
    return m_NG->getName();
}
//----------------------------------------------------------------------------
const NeuronGroup *NeuronVarReference::getNeuronGroup() const 
{ 
    return m_NG; 
}

//----------------------------------------------------------------------------
// CurrentSourceVarReference
//----------------------------------------------------------------------------
const std::string CurrentSourceVarReference::name = "CurrentSource";
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
const std::string &CurrentSourceVarReference::getTargetName() const
{
    return m_CS->getName();
}
//----------------------------------------------------------------------------
const CurrentSource *CurrentSourceVarReference::getCurrentSource() const
{
    return m_CS; 
}

//----------------------------------------------------------------------------
// PSMVarReference
//----------------------------------------------------------------------------
const std::string PSMVarReference::name = "PSM";
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
const std::string &PSMVarReference::getTargetName() const
{
    return m_SG->getPSModelTargetName();
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
const SynapseGroup *WUVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}

//----------------------------------------------------------------------------
// WUPreVarReference
//----------------------------------------------------------------------------
const std::string WUPreVarReference::name = "WUPre";
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
const std::string &WUPreVarReference::getTargetName() const
{
    return m_SG->getName();
}
//----------------------------------------------------------------------------
const SynapseGroup *WUPreVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}

//----------------------------------------------------------------------------
// WUPostVarReference
//----------------------------------------------------------------------------
const std::string WUPostVarReference::name = "WUPost";
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
const std::string &WUPostVarReference::getTargetName() const
{
    return m_SG->getName();
}
//----------------------------------------------------------------------------
const SynapseGroup *WUPostVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}