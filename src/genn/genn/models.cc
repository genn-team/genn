#include "models.h"


// GeNN includes
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

//----------------------------------------------------------------------------
// Models::VarReference
//----------------------------------------------------------------------------
Models::VarReference::VarReference(const NeuronGroup *ng, const std::string &varName)
:   m_Type(Type::Neuron), m_NG(ng), m_SG(nullptr), m_CS(nullptr)
{
    const auto *nm = ng->getNeuronModel();
    m_VarIndex = nm->getVarIndex(varName);
    m_Var = nm->getVars().at(m_VarIndex);
}
//----------------------------------------------------------------------------
Models::VarReference::VarReference(const CurrentSource *cs, const std::string &varName)
:    m_Type(Type::CurrentSource), m_NG(nullptr), m_SG(nullptr), m_CS(cs)
{
    const auto *csm = cs->getCurrentSourceModel();
    m_VarIndex = csm->getVarIndex(varName);
    m_Var = csm->getVars().at(m_VarIndex);
}
//----------------------------------------------------------------------------
Models::VarReference::VarReference(const SynapseGroup *sg, const std::string &varName, Type type)
:   m_Type(type), m_NG(nullptr), m_SG(sg), m_CS(nullptr)
{
    assert(m_Type != Type::Neuron && m_Type != Type::CurrentSource);

    if(m_Type == Type::PSM) {
        if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM)) {
            throw std::runtime_error("Only individual postsynaptic model variables can be referenced.");
        }
        const auto *psm = sg->getPSModel();
        m_VarIndex = psm->getVarIndex(varName);
        m_Var = psm->getVars().at(m_VarIndex);
    }
    else {
        const auto *wum = sg->getWUModel();

        if(m_Type == Type::WU) {
            if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
                throw std::runtime_error("Only INDIVIDUAL weight update models can be referenced.");
            }
            m_VarIndex = wum->getVarIndex(varName);
            m_Var = wum->getVars().at(m_VarIndex);
        }
        else if(m_Type == Type::WUPre) {
            m_VarIndex = wum->getPreVarIndex(varName);
            m_Var = wum->getPreVars().at(m_VarIndex);
        }
        else {
            assert(m_Type == Type::WUPost);
            m_VarIndex = wum->getPostVarIndex(varName);
            m_Var = wum->getPostVars().at(m_VarIndex);
        }
    }
}
//----------------------------------------------------------------------------
/*std::string Models::VarReference::getVarName() const
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
size_t Models::VarReference::getVarSize(const CodeGenerator::BackendBase &backend) const
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
}*/
//----------------------------------------------------------------------------
const NeuronGroup *Models::VarReference::getNeuronGroup() const
{
    assert(m_Type == Type::Neuron);
    return m_NG;
}
//----------------------------------------------------------------------------
const SynapseGroup *Models::VarReference::getSynapseGroup() const
{
    assert(m_Type != Type::Neuron && m_Type != Type::CurrentSource);
    return m_SG;
}
//----------------------------------------------------------------------------
const CurrentSource *Models::VarReference::getCurrentSource() const
{
    assert(m_Type == Type::CurrentSource);
    return m_CS;
}