#include "varReference.h"

// GeNN includes
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

//----------------------------------------------------------------------------
// VarReference
//----------------------------------------------------------------------------
std::string VarReference::getVarName() const
{
    switch(m_Type) {
    case Type::Neuron:
        return m_Var.name + m_NG->getName();
    case Type::CurrentSource:
        return m_Var.name + m_CS->getName();
    case Type::PSM:
        return m_Var.name + m_SG->getPSModelTargetName();
    case Type::WU:
    case Type::WUPre:
    case Type::WUPost:
        return m_Var.name + m_SG->getName();
    }
}
//----------------------------------------------------------------------------
size_t VarReference::getVarSize(const CodeGenerator::BackendBase &backend) const
{
    switch(m_Type) {
    case Type::Neuron:
        return m_NG->getNumNeurons();
    case Type::CurrentSource:
        return m_CS->getTrgNeuronGroup()->getNumNeurons();
    case Type::PSM:
    case Type::WUPost:
        return m_SG->getTrgNeuronGroup()->getNumNeurons();
    case Type::WU:
        return m_SG->getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(*m_SG);
    case Type::WUPre:
        return m_SG->getSrcNeuronGroup()->getNumNeurons();
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::create(const NeuronGroup *ng, const std::string &varName)
{
    const auto *nm = ng->getNeuronModel();
    const size_t varIdx = nm->getVarIndex(varName);
    return VarReference(ng, nm->getVars().at(varIdx), Type::Neuron);
}
//----------------------------------------------------------------------------
VarReference VarReference::create(const CurrentSource *cs, const std::string &varName)
{
    const auto *csm = cs->getCurrentSourceModel();
    const size_t varIdx = csm->getVarIndex(varName);
    return VarReference(cs, csm->getVars().at(varIdx), Type::CurrentSource);
}
//----------------------------------------------------------------------------
VarReference VarReference::createPSM(const SynapseGroup *sg, const std::string &varName)
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM)) {
        throw std::runtime_error("Cannot get reference to optimised out postsynaptic model variable");
    }
    const auto *psm = sg->getPSModel();
    const size_t varIdx = psm->getVarIndex(varName);
    return VarReference(sg, psm->getVars().at(varIdx), Type::PSM);
}
//----------------------------------------------------------------------------
VarReference VarReference::createWU(const SynapseGroup *sg, const std::string &varName)
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
        throw std::runtime_error("Cannot get reference to optimised out weight update model variable");
    }
    const auto *wum = sg->getWUModel();
    const size_t varIdx = wum->getVarIndex(varName);
    return VarReference(sg, wum->getVars().at(varIdx), Type::WU);
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPre(const SynapseGroup *sg, const std::string &varName)
{
    const auto *wum = sg->getWUModel();
    const size_t varIdx = wum->getPreVarIndex(varName);
    return VarReference(sg, wum->getPreVars().at(varIdx), Type::WUPre);
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPost(const SynapseGroup *sg, const std::string &varName)
{
    const auto *wum = sg->getWUModel();
    const size_t varIdx = wum->getPostVarIndex(varName);
    return VarReference(sg, wum->getPostVars().at(varIdx), Type::WUPost);
}
//----------------------------------------------------------------------------
VarReference::VarReference(const NeuronGroup *ng, Models::Base::Var var, Type type)
:   m_NG(ng), m_SG(nullptr), m_CS(nullptr), m_Var(var), m_Type(type)
{}
//----------------------------------------------------------------------------
VarReference::VarReference(const SynapseGroup *sg, Models::Base::Var var, Type type)
:   m_NG(nullptr), m_SG(static_cast<const SynapseGroupInternal*>(sg)), m_CS(nullptr), m_Var(var), m_Type(type)
{}
//----------------------------------------------------------------------------
VarReference::VarReference(const CurrentSource *cs, Models::Base::Var var, Type type)
:   m_NG(nullptr), m_SG(nullptr), m_CS(static_cast<const CurrentSourceInternal*>(cs)), m_Var(var), m_Type(type)
{}