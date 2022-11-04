#include "models.h"

// GeNN includes
#include "customConnectivityUpdateInternal.h"
#include "customUpdateInternal.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

using namespace Models;

//----------------------------------------------------------------------------
// Models::Base
//----------------------------------------------------------------------------
void Base::updateHash(boost::uuids::detail::sha1 &hash) const
{
    // Superclass
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getVars(), hash);
}
//----------------------------------------------------------------------------
void Base::validate() const
{
    // Superclass
    Snippet::Base::validate();

    Utils::validateVecNames(getVars(), "Variable");
}

//----------------------------------------------------------------------------
// VarReference
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(NeuronGroup *ng, const std::string &varName)
{
    return VarReference(static_cast<NeuronGroupInternal *>(ng), varName);
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(CurrentSource *cs, const std::string &varName)
{
    return VarReference(static_cast<CurrentSourceInternal *>(cs), varName);
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(CustomUpdate *cu, const std::string &varName)
{
    return VarReference(static_cast<CustomUpdateInternal *>(cu), varName);
}
//----------------------------------------------------------------------------
VarReference VarReference::createPreVarRef(CustomConnectivityUpdate *cu, const std::string &varName)
{
    CustomConnectivityUpdateInternal *cuInternal = static_cast<CustomConnectivityUpdateInternal*>(cu);
    const auto *cum = cuInternal->getCustomConnectivityUpdateModel();
    auto *sg = cuInternal->getSynapseGroup();
    auto getDelayNG = [sg]() { return (sg->getDelaySteps() > 0) ? sg->getSrcNeuronGroup() : nullptr; };
    return VarReference(sg->getSrcNeuronGroup()->getNumNeurons(), getDelayNG,
                        cum->getPreVarIndex(varName), cum->getPreVars(),
                        [cuInternal]() { return cuInternal->getName(); });
}
//----------------------------------------------------------------------------
VarReference VarReference::createPostVarRef(CustomConnectivityUpdate *cu, const std::string &varName)
{
    CustomConnectivityUpdateInternal *cuInternal = static_cast<CustomConnectivityUpdateInternal*>(cu);
    const auto *cum = cuInternal->getCustomConnectivityUpdateModel();
    auto *sg = cuInternal->getSynapseGroup();
    auto getDelayNG = [sg]() { return (sg->getBackPropDelaySteps() > 0) ? sg->getTrgNeuronGroup() : nullptr; };
    return VarReference(sg->getTrgNeuronGroup()->getNumNeurons(), getDelayNG,
                        cum->getPostVarIndex(varName), cum->getPostVars(),
                        [cuInternal]() { return cuInternal->getName(); });
}
//----------------------------------------------------------------------------
VarReference VarReference::createPSMVarRef(SynapseGroup *sg, const std::string &varName)
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM)) {
        throw std::runtime_error("Only individual postsynaptic model variables can be referenced.");
    }

    SynapseGroupInternal *sgInternal = static_cast<SynapseGroupInternal*>(sg);
    const auto *psm = sgInternal->getPSModel();
    return VarReference(sgInternal->getTrgNeuronGroup()->getNumNeurons(),
                        []() { return nullptr; },
                        psm->getVarIndex(varName), psm->getVars(),
                        [sgInternal]() { return sgInternal->getFusedPSVarSuffix(); });
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPreVarRef(SynapseGroup *sg, const std::string &varName)
{
    SynapseGroupInternal *sgInternal = static_cast<SynapseGroupInternal*>(sg);
    const auto *wum = sgInternal->getWUModel();
    auto getDelayNG = [sgInternal]() { return (sgInternal->getDelaySteps() > 0) ? sgInternal->getSrcNeuronGroup() : nullptr; };
    return VarReference(sgInternal->getSrcNeuronGroup()->getNumNeurons(), getDelayNG,
                        wum->getPreVarIndex(varName), wum->getPreVars(),
                        [sgInternal]() { return sgInternal->getName(); });
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPostVarRef(SynapseGroup *sg, const std::string &varName)
{
    SynapseGroupInternal *sgInternal = static_cast<SynapseGroupInternal *>(sg);
    const auto *wum = sgInternal->getWUModel();
    auto getDelayNG = [sgInternal]() { return (sgInternal->getBackPropDelaySteps() > 0) ? sgInternal->getTrgNeuronGroup() : nullptr; };
    return VarReference(sgInternal->getTrgNeuronGroup()->getNumNeurons(), getDelayNG,
                        wum->getPostVarIndex(varName), wum->getPostVars(),
                        [sgInternal]() { return sgInternal->getName(); });
}
//----------------------------------------------------------------------------
VarReference::VarReference(NeuronGroupInternal *ng, const std::string &varName)
:   VarReferenceBase(ng->getNeuronModel()->getVarIndex(varName), ng->getNeuronModel()->getVars(), [ng](){ return ng->getName(); }),
    m_Size(ng->getNumNeurons()), m_GetDelayNeuronGroup([ng, varName]() { return (ng->isDelayRequired() && ng->isVarQueueRequired(varName)) ? ng : nullptr; })
{
}
//----------------------------------------------------------------------------
VarReference::VarReference(CurrentSourceInternal *cs, const std::string &varName)
:   VarReferenceBase(cs->getCurrentSourceModel()->getVarIndex(varName), cs->getCurrentSourceModel()->getVars(), [cs]() { return cs->getName(); }),
    m_Size(cs->getTrgNeuronGroup()->getNumNeurons()), m_GetDelayNeuronGroup([]() { return nullptr; })
{

}
//----------------------------------------------------------------------------
VarReference::VarReference(CustomUpdate *cu, const std::string &varName)
:   VarReferenceBase(cu->getCustomUpdateModel()->getVarIndex(varName), cu->getCustomUpdateModel()->getVars(), [cu]() { return cu->getName(); }),
    m_Size(cu->getSize()), m_GetDelayNeuronGroup([]() { return nullptr; })
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
WUVarReference::WUVarReference(SynapseGroup *sg, const std::string &varName,
                               SynapseGroup *transposeSG, const std::string &transposeVarName)
:   VarReferenceBase(sg->getWUModel()->getVarIndex(varName), sg->getWUModel()->getVars(), [sg]() { return sg->getName(); }),
    m_SG(static_cast<SynapseGroupInternal*>(sg)), m_TransposeSG(static_cast<SynapseGroupInternal*>(transposeSG)),
    m_TransposeVarIndex((transposeSG == nullptr) ? 0 : transposeSG->getWUModel()->getVarIndex(transposeVarName)),
    m_TransposeVar((transposeSG == nullptr) ? Models::Base::Var() : transposeSG->getWUModel()->getVars().at(m_TransposeVarIndex)),
    m_GetTransposeTargetName((transposeSG == nullptr) ? GetTargetNameFn() : [transposeSG]() { return transposeSG->getName(); })
{
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && !(sg->getMatrixType() & SynapseMatrixWeight::KERNEL)) {
        throw std::runtime_error("Only INDIVIDUAL or KERNEL weight update variables can be referenced.");
    }

    if(sg->isWeightSharingSlave()) {
        throw std::runtime_error("Only weight update model variables in weight sharing master synapse group can be referenced.");
    }

    // If a transpose synapse group is specified
    if(m_TransposeSG != nullptr) {
        // Check that both tranpose and original group has individual variables
        if(!(m_TransposeSG->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) || !(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
            throw std::runtime_error("Transpose updates can only reference INDIVIDUAL weight update variables.");
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

        // Check duplicatedness of variables
        if((getVar().access & VarAccessDuplication::DUPLICATE) != (getTransposeVar().access & VarAccessDuplication::DUPLICATE)) {
            throw std::runtime_error("Transpose updates can only be performed on similarly batched variables");
        }
    }
}
//----------------------------------------------------------------------------
WUVarReference::WUVarReference(CustomUpdateWU *cu, const std::string &varName)
:   VarReferenceBase(cu->getCustomUpdateModel()->getVarIndex(varName), cu->getCustomUpdateModel()->getVars(), [cu]() { return cu->getName(); }),
    m_SG(static_cast<CustomUpdateWUInternal*>(cu)->getSynapseGroup()), m_TransposeSG(nullptr),
    m_TransposeVarIndex(0)
{

}
//----------------------------------------------------------------------------
WUVarReference::WUVarReference(CustomConnectivityUpdate *cu, const std::string &varName)
:   VarReferenceBase(cu->getCustomConnectivityUpdateModel()->getVarIndex(varName), cu->getCustomConnectivityUpdateModel()->getVars(), [cu]() { return cu->getName(); }),
    m_SG(static_cast<CustomConnectivityUpdateInternal*>(cu)->getSynapseGroup()), m_TransposeSG(nullptr),
    m_TransposeVarIndex(0)
{

}
//----------------------------------------------------------------------------
SynapseGroup *WUVarReference::getSynapseGroup() const
{
    return m_SG;
}
//----------------------------------------------------------------------------
SynapseGroup *WUVarReference::getTransposeSynapseGroup() const
{ 
    return m_TransposeSG; 
}
//----------------------------------------------------------------------------
void Models::updateHash(const Base::Var &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.name, hash);
    Utils::updateHash(v.type, hash);
    Utils::updateHash(v.access, hash);
}
//----------------------------------------------------------------------------
void Models::updateHash(const Base::VarRef &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.name, hash);
    Utils::updateHash(v.type, hash);
    Utils::updateHash(v.access, hash);
}
//----------------------------------------------------------------------------
void Models::updateHash(const VarReference &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.getTargetName(), hash);
    Utils::updateHash(v.getVarIndex(), hash);
}
//----------------------------------------------------------------------------
void Models::updateHash(const WUVarReference &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.getTargetName(), hash);
    Utils::updateHash(v.getVarIndex(), hash);

    if(v.getTransposeSynapseGroup() != nullptr) {
        Utils::updateHash(v.getTransposeTargetName(), hash);
        Utils::updateHash(v.getTransposeVarIndex(), hash);
    }
}
