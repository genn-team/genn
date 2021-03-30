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
    return VarReference(sgInternal->getTrgNeuronGroup()->getNumNeurons(),
                        []() { return nullptr; },
                        psm->getVarIndex(varName), psm->getVars(),
                        [sgInternal]() { return sgInternal->getPSModelTargetName(); });
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPreVarRef(const SynapseGroup *sg, const std::string &varName)
{
    const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal *>(sg);
    const auto *wum = sgInternal->getWUModel();
    auto getDelayNG = [sgInternal]() { return (sgInternal->getDelaySteps() > 0) ? sgInternal->getSrcNeuronGroup() : nullptr; };
    return VarReference(sgInternal->getSrcNeuronGroup()->getNumNeurons(), getDelayNG,
                        wum->getPreVarIndex(varName), wum->getPreVars(),
                        [sgInternal]() { return sgInternal->getName(); });
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPostVarRef(const SynapseGroup *sg, const std::string &varName)
{
    const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal *>(sg);
    const auto *wum = sgInternal->getWUModel();
    auto getDelayNG = [sgInternal]() { return (sgInternal->getBackPropDelaySteps() > 0) ? sgInternal->getTrgNeuronGroup() : nullptr; };
    return VarReference(sgInternal->getTrgNeuronGroup()->getNumNeurons(), getDelayNG,
                        wum->getPostVarIndex(varName), wum->getPostVars(),
                        [sgInternal]() { return sgInternal->getName(); });
}
//----------------------------------------------------------------------------
VarReference::VarReference(const NeuronGroupInternal *ng, const std::string &varName)
:   VarReferenceBase(ng->getNeuronModel()->getVarIndex(varName), ng->getNeuronModel()->getVars(), [ng](){ return ng->getName(); }),
    m_Size(ng->getNumNeurons()), m_GetDelayNeuronGroup([ng, varName]() { return (ng->isDelayRequired() && ng->isVarQueueRequired(varName)) ? ng : nullptr; })
{
}
//----------------------------------------------------------------------------
VarReference::VarReference(const CurrentSourceInternal *cs, const std::string &varName)
:   VarReferenceBase(cs->getCurrentSourceModel()->getVarIndex(varName), cs->getCurrentSourceModel()->getVars(), [cs]() { return cs->getName(); }),
    m_Size(cs->getTrgNeuronGroup()->getNumNeurons()), m_GetDelayNeuronGroup([]() { return nullptr; })
{

}
//----------------------------------------------------------------------------
VarReference::VarReference(unsigned int size, GetDelayNeuronGroupFn getDelayNeuronGroup,
                           size_t varIndex, const Models::Base::VarVec &varVec, GetTargetNameFn getTargetNameFn)
:   VarReferenceBase(varIndex, varVec, getTargetNameFn), m_Size(size), m_GetDelayNeuronGroup(getDelayNeuronGroup)
{}

//----------------------------------------------------------------------------
// WUVarReference
//----------------------------------------------------------------------------
WUVarReference::WUVarReference(const SynapseGroup *sg, const std::string &varName,
                               const SynapseGroup *transposeSG, const std::string &transposeVarName)
:   VarReferenceBase(sg->getWUModel()->getVarIndex(varName), sg->getWUModel()->getVars(), [sg]() { return sg->getName(); }),
    m_SG(static_cast<const SynapseGroupInternal*>(sg)), m_TransposeSG(static_cast<const SynapseGroupInternal*>(transposeSG)),
    m_TransposeVarIndex((transposeSG == nullptr) ? 0 : transposeSG->getWUModel()->getVarIndex(transposeVarName)),
    m_TransposeVar((transposeSG == nullptr) ? Models::Base::Var() : transposeSG->getWUModel()->getVars().at(m_TransposeVarIndex)),
    m_GetTransposeTargetName((transposeSG == nullptr) ? GetTargetNameFn() : [transposeSG]() { return transposeSG->getName(); })
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
        throw std::runtime_error("Only INDIVIDUAL weight update models can be referenced.");
    }

    if(sg->isWeightSharingSlave()) {
        throw std::runtime_error("Only weight update model variables in weight sharing master synapse group can be referenced.");
    }
    // If a transpose synapse group is specified
    if(m_TransposeSG != nullptr) {
        // Check that tranpose group also has individual variables
        if(!(m_TransposeSG->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
            throw std::runtime_error("Only INDIVIDUAL weight update models can be referenced.");
        }

        // Check that both the tranpose and main synapse groups have dense connectivity
        if(!(m_TransposeSG->getMatrixType() & SynapseMatrixConnectivity::DENSE) || !(m_SG->getMatrixType() & SynapseMatrixConnectivity::DENSE)) {
            throw std::runtime_error("Tranpose updates can only be performed on DENSE weight update model variables.");
        }

        // Check that sizes of transpose and main synapse groups match
        if((m_TransposeSG->getSrcNeuronGroup()->getNumNeurons() != m_SG->getTrgNeuronGroup()->getNumNeurons())
           || (m_TransposeSG->getTrgNeuronGroup()->getNumNeurons() != m_SG->getSrcNeuronGroup()->getNumNeurons()))
        {
            throw std::runtime_error("Transpose updates can only be performed on connections between appropriately sized neuron groups.");
        }

        // Check types
        if(getVar().type != getTransposeVar().type) {
            throw std::runtime_error("Transpose updates can only be performed on variables with the same type");
        }

        // Check duplicatedness of varibles
        if((getVar().access & VarAccessDuplication::DUPLICATE) != (getTransposeVar().access & VarAccessDuplication::DUPLICATE)) {
            throw std::runtime_error("Transpose updates can only be performed on similarly batched variables");
        }
    }
}
//----------------------------------------------------------------------------
const SynapseGroup *WUVarReference::getSynapseGroup() const 
{ 
    return m_SG; 
}
//----------------------------------------------------------------------------
const SynapseGroup *WUVarReference::getTransposeSynapseGroup() const 
{ 
    return m_TransposeSG; 
}
