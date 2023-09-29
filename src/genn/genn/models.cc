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
// GeNN::Models::Base
//----------------------------------------------------------------------------
std::vector<Base::SynapseVar> Base::getFilteredSynapseVars(const std::vector<Base::SynapseVar> &vars, bool pre, bool post) const
{
    // Copy variables into new vector if pre and post dimensions match
    std::vector<Base::SynapseVar> filteredVars;
    std::copy_if(vars.cbegin(), vars.cend(), std::back_inserter(filteredVars),
                 [pre, post](const auto &v)
                 {
                     const auto dim = getVarAccessDim(v.access);
                     return (((dim & VarAccessDim::PRE_NEURON) == pre) 
                             && ((dim & VarAccessDim::POST_NEURON) == post));
                 });
    return filteredVars;
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
            // Otherwise, if reference is to the variables of a custom connectivity update,
            // remove BATCH dimension as these are never batched
            [](const CCURef &ref)
            { 
                return clearVarAccessDim(getVarAccessDim(ref.var.access), VarAccessDim::BATCH); 
            },
            // Otherwise, use dimensionality directly
            [](const auto &ref) { return getVarAccessDim(ref.var.access); }},
        m_Detail);
}
//----------------------------------------------------------------------------
std::optional<unsigned int> VarReference::getSize() const
{
    return std::visit(
        Utils::Overload{
            [](const NGRef &ref)->std::optional<unsigned int> { return ref.group->getNumNeurons(); },
            [](const PSMRef &ref)->std::optional<unsigned int> { return ref.group->getTrgNeuronGroup()->getNumNeurons(); },
            [](const CSRef &ref)->std::optional<unsigned int> { return ref.group->getTrgNeuronGroup()->getNumNeurons(); },
            [](const auto&)->std::optional<unsigned int> 
            {
                const auto dims =  getVarDims();
                const auto *sg = getSynapseGroupInternal();
                assert(sg);
                const bool pre = (dims & VarAccessDim::PRE);
                const bool post = (dims & VarAccessDim::POST);
                if(pre && post) {
                    return std::nullopt;
                }
                else if(pre) {
                    return sg->getSrcNeuronGroup()->getNumNeurons();
                }
                else {
                    assert(post);
                    return sg->getTrgNeuronGroup()->getNumNeurons();
                }
            }},
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
            [](const WURef &ref)->NeuronGroup* {
                const auto dims = getVarAccessDim(ref.var.access);
                const bool pre = (dims & VarAccessDim::PRE);
                const bool post = (dims & VarAccessDim::POST);
                if(pre && post) {
                    return nullptr;
                }
                else if(pre) {
                    return (ref.group->getDelaySteps() > 0) ? ref.group->getSrcNeuronGroup() : nullptr;
                }
                else {
                    assert(post);
                    return (ref.group->getBackPropDelaySteps() > 0) ? ref.group->getTrgNeuronGroup() : nullptr;
                }
            },            
            [](const auto&)->NeuronGroup* { return nullptr; }},
        m_Detail);
}
//----------------------------------------------------------------------------
SynapseGroup *VarReference::getSynapseGroup() const
{
    return getSynapseGroupInternal();
}
//----------------------------------------------------------------------------
const std::string &VarReference::getTargetName() const 
{ 
    return std::visit(
        Utils::Overload{
            [](const PSMRef &ref) { return std::cref(ref.group->getFusedPSVarSuffix()); },
            [](const WURef &ref) {
                const auto dims = getVarAccessDim(ref.var.access);
                const bool pre = (dims & VarAccessDim::PRE);
                const bool post = (dims & VarAccessDim::POST);
                if(pre && post) {
                    return std::cref(ref.group->getName());;
                }
                else if(pre) {
                    return std::cref(ref.group->getFusedWUPreVarSuffix());
                }
                else {
                    assert(post);
                    return std::cref(ref.group->getFusedWUPostVarSuffix());
                }
            [](const auto &ref) { return std::cref(ref.group->getName()); }},
        m_Detail);
}
//----------------------------------------------------------------------------
std::optional<std::string> VarReference::getTransposeVarName() const
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
std::optional<Type::UnresolvedType> VarReference::getTransposeVarType() const
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
std::optional<VarAccessDim> VarReference::getTransposeVarDims() const
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
SynapseGroup *VarReference::getTransposeSynapseGroup() const
{
    return getTransposeSynapseGroupInternal();
}
//------------------------------------------------------------------------
std::optional<std::string> VarReference::getTransposeTargetName() const
{
    const auto *transposeSG = getTransposeSynapseGroup();
    if(transposeSG) {
        return transposeSG->getName();
    }
    else {
        return std::nullopt;
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
    const auto transposeVarName = getTransposeVarName();
    const auto transposeTargetName = getTransposeTargetName();
    const auto otherTransposeVarName = other.getTransposeVarName();
    const auto otherTransposeTargetName = other.getTransposeTargetName();
    return (std::tie(getVarName(), getTargetName(), transposeVarName, transposeTargetName) 
            < std::tie(other.getVarName(), other.getTargetName(), otherTransposeVarName, otherTransposeTargetName));
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
VarReference VarReference::createVarRef(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    const auto *ccum = ccu->getCustomConnectivityUpdateModel();
    return VarReference(CCURef{static_cast<CustomConnectivityUpdateInternal*>(ccu),
                               ccum->getVars()[ccum->getVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createPSMVarRef(SynapseGroup *sg, const std::string &varName)
{
    const auto *psm = sg->getPSModel();
    return VarReference(PSMRef{static_cast<SynapseGroupInternal*>(sg),
                               psm->getVars()[psm->getVarIndex(varName)]});
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUVarRef(SynapseGroup *sg, const std::string &varName,
                                          SynapseGroup *transposeSG, const std::string &transposeVarName)
{
    const auto *wum = sg->getWUModel();
    auto *sgInternal = static_cast<SynapseGroupInternal*>(sg);
    const auto var = wum->getVars()[wum->getVarIndex(varName)];
    if(transposeSG) {
        const auto dims = getVarAccessDims(var.access);
        const auto *transposeWUM = transposeSG->getWUModel();
        const auto transposeVar = transposeWUM->getVars()[transposeWUM->getVarIndex(transposeVarName)];
        const auto transposeDims = getVarAccessDims(transposeVar.access);
        if((dims & VarAccessDim::PRE_NEURON) && (dims & VarAccessDim::POST_NEURON)
           && (transposeDims & VarAccessDim::PRE_NEURON) && (transposeDims & VarAccessDim::POST_NEURON)) {
            
            return VarReference(WURef{sgInternal, static_cast<SynapseGroupInternal*>(transposeSG),
                                      var, transposeVar});
        }
        else {
            throw std::runtime_error("Transpose variable references can only be made to synaptic variables.");
        }
    }
    else {
        return VarReference(WURef{static_cast<SynapseGroupInternal*>(sg), nullptr,
                                  var, std::nullopt});
    }
}
//----------------------------------------------------------------------------
VarReference::VarReference(const DetailType &detail) 
:   m_Detail(detail)
{
    // If reference is to a synaptic variable
    const auto varDims = getVarDims();
    if((varDims & VarAccessDim::PRE_NEURON) && (varDims & VarAccessDim::POST_NEURON)) {
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
}
//------------------------------------------------------------------------
SynapseGroupInternal *VarReference::getSynapseGroupInternal() const
{
    return std::visit(
        Utils::Overload{
            [](const PSMRef &ref) { return ref.group; },
            [](const WURef &ref) { return ref.group; },
            [](const CURef &ref) { return ref.group->getSynapseGroup(); },
            [](const CCURef &ref) { return ref.group->getSynapseGroup(); },
            [](const auto&)->SynapseGroupInternal*{ return nullptr; }},
        m_Detail);
}
//------------------------------------------------------------------------
SynapseGroupInternal *VarReference::getTransposeSynapseGroupInternal() const
{
    return std::visit(
        Utils::Overload{
            [](const WURef &ref)->SynapseGroupInternal* { return ref.transposeGroup; },
            [](const auto&)->SynapseGroupInternal* { return nullptr; }},
        m_Detail);
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
