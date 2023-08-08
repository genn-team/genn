#include "models.h"

// GeNN includes
#include "customConnectivityUpdateInternal.h"
#include "customUpdateInternal.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// GeNN::Models::Base
//----------------------------------------------------------------------------
namespace GeNN::Models
{
void Base::updateHash(boost::uuids::detail::sha1 &hash) const
{
    // Superclass
    Snippet::Base::updateHash(hash);

    Utils::updateHash(getVars(), hash);
}
//----------------------------------------------------------------------------
void Base::validate(const std::unordered_map<std::string, double> &paramValues, 
                    const std::unordered_map<std::string, InitVarSnippet::Init> &varValues,
                    const std::string &description) const
{
    // Superclass
    Snippet::Base::validate(paramValues, description);

    const auto vars = getVars();
    Utils::validateVecNames(vars, "Variable");

    // Validate variable initialisers
    Utils::validateInitialisers(vars, varValues, "variable", description);
}

//----------------------------------------------------------------------------
// VarReference
//----------------------------------------------------------------------------
NeuronGroup *VarReference::getDelayNeuronGroup() const
{ 
    return std::visit(
        Utils::Overload{
            [this](const NGRef &ref)->NeuronGroup* {
                return (ref.group->isDelayRequired() && ref.group->isVarQueueRequired(getVar().name)) ? ref.group : nullptr;
            },
            [](const WUPreRef &ref)->NeuronGroup* {
                return (ref.group->getDelaySteps() > 0) ? ref.group->getSrcNeuronGroup() : nullptr;
            },
            [](const WUPostRef &ref)->NeuronGroup* {
                return (ref.group->getBackPropDelaySteps() > 0) ? ref.group->getTrgNeuronGroup() : nullptr;
            },
            [](const auto&)->NeuronGroup* { return nullptr; }},
        m_Detail);
}
//----------------------------------------------------------------------------
std::string VarReference::getTargetName() const 
{ 
    return std::visit(
            Utils::Overload{
            [](const PSMRef &ref) { return ref.group->getFusedPSVarSuffix(); },
            [](const WUPreRef &ref) { return ref.group->getFusedWUPreVarSuffix(); },
            [](const WUPostRef &ref) { return ref.group->getFusedWUPostVarSuffix(); },
            [](const auto &ref) { return ref.group->getName(); }},
        m_Detail);
}
//----------------------------------------------------------------------------
bool VarReference::isDuplicated() const
{
    if(getVar().access & VarAccessDuplication::SHARED) {
        return false;
    }
    else {
        return std::visit(
            Utils::Overload{
                [](const CURef &ref) { return ref.group->isBatched(); },
                [](const auto&) { return true; }},
            m_Detail);
    }
}
//----------------------------------------------------------------------------
CustomUpdate *VarReference::getReferencedCustomUpdate() const
{
    return std::visit(
            Utils::Overload{
                [](const CURef &ref)->CustomUpdate* { return ref.group; },
                [](const auto&)->CustomUpdate* { return nullptr; }},
            m_Detail);
}
//----------------------------------------------------------------------------
bool VarReference::operator < (const VarReference &other) const
{
    // **NOTE** variable and target names are enough to guarantee uniqueness
    const std::string targetName = getTargetName();
    const std::string otherTargetName = other.getTargetName();
    return (std::tie(getVar().name, targetName) < std::tie(other.getVar().name, otherTargetName));
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(NeuronGroup *ng, const std::string &varName)
{
    return VarReference(ng->getNeuronModel()->getVarIndex(varName), ng->getNeuronModel()->getVars(),
                        ng->getNumNeurons(), NGRef{static_cast<NeuronGroupInternal*>(ng)});
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(CurrentSource *cs, const std::string &varName)
{
    auto *csInternal = static_cast<CurrentSourceInternal*>(cs);
    return VarReference(cs->getCurrentSourceModel()->getVarIndex(varName), cs->getCurrentSourceModel()->getVars(),
                        csInternal->getTrgNeuronGroup()->getNumNeurons(), CSRef{csInternal});
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(CustomUpdate *cu, const std::string &varName)
{
    return VarReference(cu->getCustomUpdateModel()->getVarIndex(varName), cu->getCustomUpdateModel()->getVars(),
                        cu->getSize(), CURef{static_cast<CustomUpdateInternal*>(cu)});
}
//----------------------------------------------------------------------------
VarReference VarReference::createPreVarRef(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    auto *ccuInternal = static_cast<CustomConnectivityUpdateInternal*>(ccu);
    auto *sg = ccuInternal->getSynapseGroup();
    return VarReference(ccu->getCustomConnectivityUpdateModel()->getPreVarIndex(varName), ccu->getCustomConnectivityUpdateModel()->getPreVars(),
                        sg->getSrcNeuronGroup()->getNumNeurons(), CCUPreRef{ccuInternal});
}
//----------------------------------------------------------------------------
VarReference VarReference::createPostVarRef(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    auto *ccuInternal = static_cast<CustomConnectivityUpdateInternal*>(ccu);
    auto *sg = ccuInternal->getSynapseGroup();
    return VarReference(ccu->getCustomConnectivityUpdateModel()->getPostVarIndex(varName), ccu->getCustomConnectivityUpdateModel()->getPostVars(),
                        sg->getTrgNeuronGroup()->getNumNeurons(), CCUPostRef{ccuInternal});
}
//----------------------------------------------------------------------------
VarReference VarReference::createPSMVarRef(SynapseGroup *sg, const std::string &varName)
{
    auto *sgInternal = static_cast<SynapseGroupInternal*>(sg);
    return VarReference(sg->getPSModel()->getVarIndex(varName), sg->getPSModel()->getVars(),
                        sgInternal->getTrgNeuronGroup()->getNumNeurons(), PSMRef{sgInternal});
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPreVarRef(SynapseGroup *sg, const std::string &varName)
{
    auto *sgInternal = static_cast<SynapseGroupInternal*>(sg);
    return VarReference(sg->getWUModel()->getPreVarIndex(varName), sg->getWUModel()->getPreVars(),
                        sgInternal->getSrcNeuronGroup()->getNumNeurons(), WUPreRef{sgInternal});
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPostVarRef(SynapseGroup *sg, const std::string &varName)
{
    auto *sgInternal = static_cast<SynapseGroupInternal*>(sg);
    return VarReference(sg->getWUModel()->getPostVarIndex(varName), sg->getWUModel()->getPostVars(),
                        sgInternal->getTrgNeuronGroup()->getNumNeurons(), WUPostRef{sgInternal});
}

//----------------------------------------------------------------------------
// WUVarReference
//----------------------------------------------------------------------------
std::string WUVarReference::getTargetName() const
{
    return std::visit(
        Utils::Overload{
            [](const auto &ref) { return ref.group->getName(); }},
        m_Detail);
}
//----------------------------------------------------------------------------
bool WUVarReference::isDuplicated() const
{
    if(getVar().access & VarAccessDuplication::SHARED) {
        return false;
    }
    else {
        return std::visit(
            Utils::Overload{
                [](const CURef &ref) { return ref.group->isBatched(); },
                [](const auto&) { return true; }},
            m_Detail);
    }
}
//----------------------------------------------------------------------------
SynapseGroup *WUVarReference::getSynapseGroup() const
{
    return getSynapseGroupInternal();
}
//------------------------------------------------------------------------
SynapseGroup *WUVarReference::getTransposeSynapseGroup() const
{
    return getTransposeSynapseGroupInternal();
}
//------------------------------------------------------------------------
std::string WUVarReference::getTransposeTargetName() const
{
    return std::visit(
        Utils::Overload{
            [](const WURef &ref) { return ref.transposeGroup->getName(); },
            [](const auto&)->std::string { throw std::runtime_error("No transpose"); }},
        m_Detail);
}
//------------------------------------------------------------------------
CustomUpdateWU *WUVarReference::getReferencedCustomUpdate() const
{
    return std::visit(
        Utils::Overload{
            [](const CURef &ref)->CustomUpdateWU* { return ref.group; },
            [](const auto&)->CustomUpdateWU* { return nullptr; }},
        m_Detail);
}
//------------------------------------------------------------------------
bool WUVarReference::operator < (const WUVarReference &other) const
{
    // **NOTE** variable and target names are enough to guarantee uniqueness
    const bool hasTranspose = (getTransposeSynapseGroup() != nullptr);
    const bool otherHasTranspose = (other.getTransposeSynapseGroup() != nullptr);
    if (hasTranspose && otherHasTranspose) {
        return (std::make_tuple(getVar().name, getTargetName(), getTransposeVar().name, getTransposeTargetName()) 
                < std::tuple(other.getVar().name, other.getTargetName(), other.getTransposeVar().name, other.getTransposeTargetName()));
    }
    else if (hasTranspose) {
        return false;
    }
    else if (otherHasTranspose) {
        return true;
    }
    else {
        return (std::make_tuple(getVar().name, getTargetName()) 
                < std::make_tuple(other.getVar().name, other.getTargetName()));
    }
}
//------------------------------------------------------------------------
WUVarReference WUVarReference::createWUVarReference(SynapseGroup *sg, const std::string &varName, 
                                                    SynapseGroup *transposeSG, const std::string &transposeVarName)
{
    if(transposeSG) {
        return WUVarReference(sg->getWUModel()->getVarIndex(varName), sg->getWUModel()->getVars(), 
                              transposeSG->getWUModel()->getVarIndex(transposeVarName), transposeSG->getWUModel()->getVars(),
                              WURef{static_cast<SynapseGroupInternal*>(sg), static_cast<SynapseGroupInternal*>(transposeSG)});
    }
    else {
        return WUVarReference(sg->getWUModel()->getVarIndex(varName), sg->getWUModel()->getVars(),
                              WURef{static_cast<SynapseGroupInternal*>(sg), static_cast<SynapseGroupInternal*>(transposeSG)});
    }
}
//------------------------------------------------------------------------
WUVarReference WUVarReference::createWUVarReference(CustomUpdateWU *cu, const std::string &varName)
{
    return WUVarReference(cu->getCustomUpdateModel()->getVarIndex(varName), cu->getCustomUpdateModel()->getVars(),
                          CURef{static_cast<CustomUpdateWUInternal*>(cu)});
}
//------------------------------------------------------------------------
WUVarReference WUVarReference::createWUVarReference(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    return WUVarReference(ccu->getCustomConnectivityUpdateModel()->getVarIndex(varName), ccu->getCustomConnectivityUpdateModel()->getVars(),
                          CCURef{static_cast<CustomConnectivityUpdateInternal*>(ccu)});
}
//------------------------------------------------------------------------
SynapseGroupInternal *WUVarReference::getSynapseGroupInternal() const
{
    return std::visit(
        Utils::Overload{
            [](const WURef &ref) { return ref.group; },
            [](const auto &ref) { return ref.group->getSynapseGroup(); }},
        m_Detail);
}
//------------------------------------------------------------------------
SynapseGroupInternal *WUVarReference::getTransposeSynapseGroupInternal() const
{
    return std::visit(
        Utils::Overload{
            [](const WURef &ref)->SynapseGroupInternal* { return ref.transposeGroup; },
            [](const auto&)->SynapseGroupInternal* { return nullptr; }},
        m_Detail);
}
//------------------------------------------------------------------------
WUVarReference::WUVarReference(size_t varIndex, const Models::Base::VarVec &varVec,
                               const DetailType &detail)
:   VarReferenceBase(varIndex, varVec), m_TransposeVarIndex(std::nullopt), 
    m_TransposeVar(std::nullopt), m_Detail(detail)
{
    // Check matrix types
    auto *sg = getSynapseGroup();
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && !(sg->getMatrixType() & SynapseMatrixWeight::KERNEL)) {
        throw std::runtime_error("Only INDIVIDUAL or KERNEL weight update variables can be referenced.");
    }
}
//------------------------------------------------------------------------
WUVarReference::WUVarReference(size_t varIndex, const Models::Base::VarVec &varVec,
                               size_t transposeVarIndex, const Models::Base::VarVec &transposeVarVec,
                               const DetailType &detail)
:   VarReferenceBase(varIndex, varVec), m_TransposeVarIndex(transposeVarIndex), 
    m_TransposeVar(transposeVarVec.at(transposeVarIndex)), m_Detail(detail)
{
    // Check matrix types
    auto *sg = getSynapseGroupInternal();
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && !(sg->getMatrixType() & SynapseMatrixWeight::KERNEL)) {
        throw std::runtime_error("Only INDIVIDUAL or KERNEL weight update variables can be referenced.");
    }

    // Check that both tranpose and original group has individual variables
    auto *transposeSG = getTransposeSynapseGroupInternal();
    if(!(transposeSG->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) || !(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL)) {
        throw std::runtime_error("Transpose updates can only reference INDIVIDUAL weight update variables.");
    }

    // Check that both the tranpose and main synapse groups have dense connectivity
    if(!(transposeSG->getMatrixType() & SynapseMatrixConnectivity::DENSE) || !(sg->getMatrixType() & SynapseMatrixConnectivity::DENSE)) {
        throw std::runtime_error("Tranpose updates can only be performed on DENSE weight update model variables.");
    }

    // Check that sizes of transpose and main synapse groups match
    if((transposeSG->getSrcNeuronGroup()->getNumNeurons() != sg->getTrgNeuronGroup()->getNumNeurons())
        || (transposeSG->getTrgNeuronGroup()->getNumNeurons() != sg->getSrcNeuronGroup()->getNumNeurons()))
    {
        throw std::runtime_error("Transpose updates can only be performed on connections between appropriately sized neuron groups.");
    }

    // Check types
    // **NOTE** this is a bit over-conservative as, at this point, types are not resolved so "scalar" cannot be compared with "float"
    if(getVar().type != getTransposeVar().type) {
        throw std::runtime_error("Transpose updates can only be performed on variables with the same type");
    }

    // Check duplicatedness of variables
    if((getVar().access & VarAccessDuplication::DUPLICATE) != (getTransposeVar().access & VarAccessDuplication::DUPLICATE)) {
        throw std::runtime_error("Transpose updates can only be performed on similarly batched variables");
    }
}

//----------------------------------------------------------------------------
// EGPReference
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(const NeuronGroup *ng, const std::string &egpName)
{
    const auto *nm = ng->getNeuronModel();
    return EGPReference(nm->getExtraGlobalParamIndex(egpName), nm->getExtraGlobalParams(), ng->getName());
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(const CurrentSource *cs, const std::string &egpName)
{
    const auto *cm = cs->getCurrentSourceModel();
    return EGPReference(cm->getExtraGlobalParamIndex(egpName), cm->getExtraGlobalParams(), cs->getName());
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(const CustomUpdate *cu, const std::string &egpName)
{
    const auto *cm = cu->getCustomUpdateModel();
    return EGPReference(cm->getExtraGlobalParamIndex(egpName), cm->getExtraGlobalParams(), cu->getName());
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(const CustomUpdateWU *cu, const std::string &egpName)
{
    const auto *cm = cu->getCustomUpdateModel();
    return EGPReference(cm->getExtraGlobalParamIndex(egpName), cm->getExtraGlobalParams(), cu->getName());
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createPSMEGPRef(const SynapseGroup *sg, const std::string &egpName)
{
    const auto *psm = sg->getPSModel();
    return EGPReference(psm->getExtraGlobalParamIndex(egpName), psm->getExtraGlobalParams(), sg->getName());
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createWUEGPRef(const SynapseGroup *sg, const std::string &egpName)
{
    const auto *wum = sg->getWUModel();
    return EGPReference(wum->getExtraGlobalParamIndex(egpName), wum->getExtraGlobalParams(), sg->getName());
}

//----------------------------------------------------------------------------
// Free functions
//----------------------------------------------------------------------------
void updateHash(const Base::Var &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.name, hash);
    Type::updateHash(v.type, hash);
    Utils::updateHash(v.access, hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::VarRef &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.name, hash);
    Type::updateHash(v.type, hash);
    Utils::updateHash(v.access, hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::EGPRef &e, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(e.name, hash);
    Type::updateHash(e.type, hash);
}
//----------------------------------------------------------------------------
void updateHash(const VarReference &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.getTargetName(), hash);
    Utils::updateHash(v.getVarIndex(), hash);
}
//----------------------------------------------------------------------------
void updateHash(const WUVarReference &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.getTargetName(), hash);
    Utils::updateHash(v.getVarIndex(), hash);

    if(v.getTransposeSynapseGroup() != nullptr) {
        Utils::updateHash(v.getTransposeTargetName(), hash);
        Utils::updateHash(v.getTransposeVarIndex(), hash);
    }
}
//----------------------------------------------------------------------------
void updateHash(const EGPReference &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.getTargetName(), hash);
    Utils::updateHash(v.getEGPIndex(), hash);
}
}   // namespace GeNN::Models
