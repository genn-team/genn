#include "models.h"

// GeNN includes
#include "customConnectivityUpdateInternal.h"
#include "customUpdateInternal.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

//----------------------------------------------------------------------------
// GeNN::Models::Base::EGPRef
//----------------------------------------------------------------------------
namespace GeNN::Models
{
Base::EGPRef::EGPRef(const std::string &n, const std::string &t) 
:   name(n), type(Utils::handleLegacyEGPType(t))
{
}

//----------------------------------------------------------------------------
// VarReference
//----------------------------------------------------------------------------
const std::string &VarReference::getVarName() const
{
    return std::visit(
        Utils::Overload{[](const auto &ref){ return std::cref(ref.var.name); }},
        m_Detail);
}
//----------------------------------------------------------------------------
const Type::UnresolvedType &VarReference::getVarType() const
{
    return std::visit(
        Utils::Overload{[](const auto &ref){ return std::cref(ref.var.type); }},
        m_Detail);
}
//----------------------------------------------------------------------------
VarAccessDim VarReference::getVarDims() const
{
    return std::visit(
        Utils::Overload{
            // If reference is to a custom update variable, 
            // remove dimensions from those of update
            [](const CURef &ref) 
            { 
                return getVarAccessDim(ref.var.access, ref.group->getDims());
            },
            // Otherwise, if reference is to the presynaptic variables of a custom connectivity update,
            // remove BATCH dimension as these are never batched
            [](const CCUPreRef &ref)
            { 
                return clearVarAccessDim(getVarAccessDim(ref.var.access), VarAccessDim::BATCH); 
            },
            // Otherwise, if reference is to the postsynaptic variables of a custom connectivity update,
            // remove BATCH dimension as these are never batched
            [](const CCUPostRef &ref)
            { 
                return clearVarAccessDim(getVarAccessDim(ref.var.access), VarAccessDim::BATCH); 
            },
            // Otherwise, use dimensionality directly
            [](const auto &ref) { return getVarAccessDim(ref.var.access); }},
        m_Detail);
}
//----------------------------------------------------------------------------
unsigned int VarReference::getSize() const
{
    return std::visit(
            Utils::Overload{
            [](const NGRef &ref) { return ref.group->getNumNeurons(); },
            [](const PSMRef &ref) { return ref.group->getTrgNeuronGroup()->getNumNeurons(); },
            [](const WUPreRef &ref) { return ref.group->getSrcNeuronGroup()->getNumNeurons(); },
            [](const WUPostRef &ref) { return ref.group->getTrgNeuronGroup()->getNumNeurons(); },
            [](const CSRef &ref) { return ref.group->getTrgNeuronGroup()->getNumNeurons(); },
            [](const CURef &ref) { return ref.group->getSize(); },
            [](const CCUPreRef &ref) { return ref.group->getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); },
            [](const CCUPostRef &ref) { return ref.group->getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(); }},
        m_Detail);
}
//----------------------------------------------------------------------------
NeuronGroup *VarReference::getDelayNeuronGroup() const
{ 
    return std::visit(
        Utils::Overload{
            [this](const NGRef &ref)->NeuronGroup* {
                return (ref.group->isDelayRequired() && ref.group->isVarQueueRequired(ref.var.name)) ? ref.group : nullptr;
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
const std::string &VarReference::getTargetName() const 
{ 
    return std::visit(
        Utils::Overload{
            [](const PSMRef &ref) { return std::cref(ref.group->getFusedPSVarSuffix()); },
            [](const WUPreRef &ref) { return std::cref(ref.group->getFusedWUPreVarSuffix()); },
            [](const WUPostRef &ref) { return std::cref(ref.group->getFusedWUPostVarSuffix()); },
            [](const auto &ref) { return std::cref(ref.group->getName()); }},
        m_Detail);
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

    return std::visit(
        Utils::Overload{
            [&targetName, &otherTargetName](const auto &detail, const auto &otherDetail)
            { 
                return (std::tie(detail.var.name, targetName) 
                        < std::tie(otherDetail.var.name, otherTargetName)); 
            }},
            m_Detail, other.m_Detail);
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(NeuronGroup *ng, const std::string &varName)
{
    const auto *nm = ng->getNeuronModel();
    return VarReference(NGRef{static_cast<NeuronGroupInternal*>(ng), 
                              nm->getVars()[nm->getVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(CurrentSource *cs, const std::string &varName)
{
    const auto *csm = cs->getCurrentSourceModel();
    return VarReference(CSRef{static_cast<CurrentSourceInternal*>(cs),
                              csm->getVars()[csm->getVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(CustomUpdate *cu, const std::string &varName)
{
    const auto *cum = cu->getCustomUpdateModel();
    return VarReference(CURef{static_cast<CustomUpdateInternal*>(cu),
                              cum->getVars()[cum->getVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createPreVarRef(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    const auto *ccum = ccu->getCustomConnectivityUpdateModel();
    return VarReference(CCUPreRef{static_cast<CustomConnectivityUpdateInternal*>(ccu),
                                  ccum->getPreVars()[ccum->getPreVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createPostVarRef(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    const auto *ccum = ccu->getCustomConnectivityUpdateModel();
    return VarReference(CCUPostRef{static_cast<CustomConnectivityUpdateInternal*>(ccu),
                                   ccum->getPostVars()[ccum->getPostVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createPSMVarRef(SynapseGroup *sg, const std::string &varName)
{
    const auto *psm = sg->getPSModel();
    return VarReference(PSMRef{static_cast<SynapseGroupInternal*>(sg),
                               psm->getVars()[psm->getVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPreVarRef(SynapseGroup *sg, const std::string &varName)
{
    const auto *wum = sg->getWUModel();
    return VarReference(WUPreRef{static_cast<SynapseGroupInternal*>(sg),
                                 wum->getPreVars()[wum->getPreVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPostVarRef(SynapseGroup *sg, const std::string &varName)
{
    const auto *wum = sg->getWUModel();
    return VarReference(WUPostRef{static_cast<SynapseGroupInternal*>(sg),
                                  wum->getPostVars()[wum->getPostVarIndex(varName)]});
}

//----------------------------------------------------------------------------
// WUVarReference
//----------------------------------------------------------------------------
const std::string &WUVarReference::getVarName() const
{
    return std::visit(
        Utils::Overload{[](const auto &ref){ return std::cref(ref.var.name); }},
        m_Detail);
}
//----------------------------------------------------------------------------
const Type::UnresolvedType &WUVarReference::getVarType() const
{
    return std::visit(
        Utils::Overload{[](const auto &ref){ return std::cref(ref.var.type); }},
        m_Detail);
}
//----------------------------------------------------------------------------
VarAccessDim WUVarReference::getVarDims() const
{
    return std::visit(
        Utils::Overload{
            // If reference is to a custom update variable, 
            // remove dimensions from those of update
            [](const CURef &ref) 
            { 
                return getVarAccessDim(ref.var.access, ref.group->getDims());
            },
            // Otherwise, if reference is to the synaptic variables of a custom connectivity update,
            // remove BATCH dimension as these are never batched
            [](const CCURef &ref) 
            { 
                return clearVarAccessDim(getVarAccessDim(ref.var.access), VarAccessDim::BATCH); 
            },
            // Otherwise, use dimensionality directly
            [](const WURef &ref){ return getVarAccessDim(ref.var.access); }},
        m_Detail);
}
//----------------------------------------------------------------------------
const std::string &WUVarReference::getTargetName() const
{
    return std::visit(
        Utils::Overload{[](const auto &ref) { return std::cref(ref.group->getName()); }},
        m_Detail);
}
//----------------------------------------------------------------------------
SynapseGroup *WUVarReference::getSynapseGroup() const
{
    return getSynapseGroupInternal();
}
//------------------------------------------------------------------------
std::optional<std::string> WUVarReference::getTransposeVarName() const
{
    return std::visit(
        Utils::Overload{
            [](const WURef &ref)->std::optional<std::string>
            { 
                if(ref.transposeVar) {
                    return ref.transposeVar->name;
                }
                else {
                    return std::nullopt;
                }
            },
            [](const auto&)->std::optional<std::string>{ return std::nullopt; }},
        m_Detail);
}
//------------------------------------------------------------------------
std::optional<Type::UnresolvedType> WUVarReference::getTransposeVarType() const
{
    return std::visit(
        Utils::Overload{
            [](const WURef &ref)->std::optional<Type::UnresolvedType>
            { 
                if(ref.transposeVar) {
                    return ref.transposeVar->type;
                }
                else {
                    return std::nullopt;
                }
            },
            [](const auto&)->std::optional<Type::UnresolvedType>{ return std::nullopt; }},
        m_Detail);
}
//------------------------------------------------------------------------
std::optional<VarAccessDim> WUVarReference::getTransposeVarDims() const
{
    return std::visit(
        Utils::Overload{
            [](const WURef &ref)->std::optional<VarAccessDim>
            { 
                if(ref.transposeVar) {
                    return getVarAccessDim(ref.transposeVar->access);
                }
                else {
                    return std::nullopt;
                }
            },
            [](const auto&)->std::optional<VarAccessDim>{ return std::nullopt; }},
        m_Detail);
}
//------------------------------------------------------------------------
SynapseGroup *WUVarReference::getTransposeSynapseGroup() const
{
    return getTransposeSynapseGroupInternal();
}
//------------------------------------------------------------------------
std::optional<std::string> WUVarReference::getTransposeTargetName() const
{
    const auto *transposeSG = getTransposeSynapseGroup();
    if(transposeSG) {
        return transposeSG->getName();
    }
    else {
        return std::nullopt;
    }
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
    const auto transposeVarName = getTransposeVarName();
    const auto transposeTargetName = getTransposeTargetName();
    const auto otherTransposeVarName = other.getTransposeVarName();
    const auto otherTransposeTargetName = other.getTransposeTargetName();
    return (std::tie(getVarName(), getTargetName(), transposeVarName, transposeTargetName) 
            < std::tie(other.getVarName(), other.getTargetName(), otherTransposeVarName, otherTransposeTargetName));
}
//------------------------------------------------------------------------
WUVarReference WUVarReference::createWUVarReference(SynapseGroup *sg, const std::string &varName, 
                                                    SynapseGroup *transposeSG, const std::string &transposeVarName)
{
    const auto *wum = sg->getWUModel();
    auto *sgInternal = static_cast<SynapseGroupInternal*>(sg);
    const auto var = wum->getVars()[wum->getVarIndex(varName)];
    if(transposeSG) {
        const auto *transposeWUM = transposeSG->getWUModel();
        return WUVarReference(WURef{sgInternal, static_cast<SynapseGroupInternal*>(transposeSG),
                                    var, transposeWUM->getVars()[transposeWUM->getVarIndex(transposeVarName)]});
    }
    else {
        return WUVarReference(WURef{static_cast<SynapseGroupInternal*>(sg), nullptr,
                                    var, std::nullopt});
    }
}
//------------------------------------------------------------------------
WUVarReference WUVarReference::createWUVarReference(CustomUpdateWU *cu, const std::string &varName)
{
    const auto *cum = cu->getCustomUpdateModel();
    return WUVarReference(CURef{static_cast<CustomUpdateWUInternal*>(cu),
                                cum->getVars()[cum->getVarIndex(varName)]});
}
//------------------------------------------------------------------------
WUVarReference WUVarReference::createWUVarReference(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    const auto *ccum = ccu->getCustomConnectivityUpdateModel();
    return WUVarReference(CCURef{static_cast<CustomConnectivityUpdateInternal*>(ccu),
                                 ccum->getVars()[ccum->getVarIndex(varName)]});
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
WUVarReference::WUVarReference(const DetailType &detail)
:   m_Detail(detail)
{
    // Check matrix types
    auto *sg = getSynapseGroupInternal();
    if(!(sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && !(sg->getMatrixType() & SynapseMatrixWeight::KERNEL)) {
        throw std::runtime_error("Only INDIVIDUAL or KERNEL weight update variables can be referenced.");
    }

    // Check that both tranpose and original group has individual variables
    auto *transposeSG = getTransposeSynapseGroupInternal();
    if(transposeSG) {
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
        if(getVarType() != getTransposeVarType()) {
            throw std::runtime_error("Transpose updates can only be performed on variables with the same type");
        }

        // Check duplicatedness of variables
        if((getVarDims() & VarAccessDim::BATCH) 
           != (*getTransposeVarDims() & VarAccessDim::BATCH)) 
        {
            throw std::runtime_error("Transpose updates can only be performed on similarly batched variables");
        }
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
void updateHash(const Base::NeuronVar &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.name, hash);
    Type::updateHash(v.type, hash);
    Utils::updateHash(v.access, hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::SynapseVar &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.name, hash);
    Type::updateHash(v.type, hash);
    Utils::updateHash(v.access, hash);
}
//----------------------------------------------------------------------------
void updateHash(const Base::CustomUpdateVar &v, boost::uuids::detail::sha1 &hash)
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
    Utils::updateHash(v.getVarName(), hash);
}
//----------------------------------------------------------------------------
void updateHash(const WUVarReference &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.getTargetName(), hash);
    Utils::updateHash(v.getVarName(), hash);

    if(v.getTransposeSynapseGroup() != nullptr) {
        Utils::updateHash(v.getTransposeTargetName(), hash);
        Utils::updateHash(v.getTransposeVarName(), hash);
    }
}
//----------------------------------------------------------------------------
void updateHash(const EGPReference &v, boost::uuids::detail::sha1 &hash)
{
    Utils::updateHash(v.getTargetName(), hash);
    Utils::updateHash(v.getEGPIndex(), hash);
}
}   // namespace GeNN::Models
