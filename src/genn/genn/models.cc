#include "models.h"


// GeNN includes
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

using namespace Models;

//----------------------------------------------------------------------------
// VarReference
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(const NeuronGroup *ng, const std::string &varName)
{
    return VarReference(static_cast<const NeuronGroupInternal *>(ng), varName);
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(const CurrentSource *cs, const std::string &varName)
{
    return VarReference(static_cast<const CurrentSourceInternal *>(cs), varName);
}
//----------------------------------------------------------------------------
VarReference VarReference::createPSMVarRef(const SynapseGroup *sg, const std::string &varName)
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM)) {
        throw std::runtime_error("Only individual postsynaptic model variables can be referenced.");
    }

    const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal *>(sg);
    const auto *psm = sgInternal->getPSModel();
    return VarReference([sgInternal]() { return sgInternal->getPSModelTargetName(); },
                        sgInternal->getTrgNeuronGroup()->getNumNeurons(), 
                        psm->getVarIndex(varName), psm->getVars());

}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPreVarRef(const SynapseGroup *sg, const std::string &varName)
{
    const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal *>(sg);
    const auto *wum = sgInternal->getWUModel();
    return VarReference([sgInternal]() { return sgInternal->getName(); },
                        sgInternal->getSrcNeuronGroup()->getNumNeurons(), 
                        wum->getPreVarIndex(varName), wum->getPreVars());
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPostVarRef(const SynapseGroup *sg, const std::string &varName)
{
    const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal *>(sg);
    const auto *wum = sgInternal->getWUModel();
    return VarReference([sgInternal]() { return sgInternal->getName(); },
                        sgInternal->getTrgNeuronGroup()->getNumNeurons(), 
                        wum->getPostVarIndex(varName), wum->getPostVars());
}
//----------------------------------------------------------------------------
VarReference::VarReference(const NeuronGroupInternal *ng, const std::string &varName)
:   VarReferenceBase(ng->getNeuronModel()->getVarIndex(varName), ng->getNeuronModel()->getVars(), [ng](){ return ng->getName(); }),
    m_Size(ng->getNumNeurons())
{

}
//----------------------------------------------------------------------------
VarReference::VarReference(const CurrentSourceInternal *cs, const std::string &varName)
:   VarReferenceBase(cs->getCurrentSourceModel()->getVarIndex(varName), cs->getCurrentSourceModel()->getVars(), [cs]() { return cs->getName(); }),
    m_Size(cs->getTrgNeuronGroup()->getNumNeurons())
{

}
//----------------------------------------------------------------------------
VarReference::VarReference(GetTargetNameFn getTargetNameFn, unsigned int size, 
                           size_t varIndex, const Models::Base::VarVec &varVec)
:   VarReferenceBase(varIndex, varVec, getTargetNameFn), m_Size(size)
{}

//----------------------------------------------------------------------------
// WUVarReference
//----------------------------------------------------------------------------
WUVarReference::WUVarReference(const SynapseGroup *sg, const std::string &varName)
:   VarReferenceBase(sg->getWUModel()->getVarIndex(varName), sg->getWUModel()->getVars(), [sg]() { return sg->getName(); }),
    m_SG(static_cast<const SynapseGroupInternal*>(sg))
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
        throw std::runtime_error("Only INDIVIDUAL weight update models can be referenced.");
    }
}
//----------------------------------------------------------------------------
const SynapseGroup *WUVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}