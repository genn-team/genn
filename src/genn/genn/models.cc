#include "models.h"

// GeNN includes
#include "customConnectivityUpdateInternal.h"
#include "customUpdateInternal.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"

// GeNN runtime includes
#include "runtime/runtime.h"

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
Type::UnresolvedType VarReference::getVarType() const
{
    return std::visit(
        Utils::Overload{
            [](const InternalRef &ref)
            {
                return Type::UnresolvedType("scalar");
            },
            [](const auto &ref){ return ref.var.type; }},
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
            // Otherwise, if reference is internal
            [](const InternalRef &ref) 
            {
                return static_cast<VarAccessDim>(static_cast<unsigned int>(VarAccessDim::ELEMENT) | static_cast<unsigned int>(VarAccessDim::BATCH));
            },
            // Otherwise, use dimensionality directly
            [](const auto &ref) { return getVarAccessDim(ref.var.access); }},
        m_Detail);
}
//----------------------------------------------------------------------------
std::string VarReference::getVarName() const
{
    return std::visit(
        Utils::Overload{
            [](const InternalRef &ref) -> std::string
            {
                if(ref.type == InternalRef::Type::OUT_POST) {
                    return "outPost";
                }
                else {
                    return "outPre";
                }
            },
            [](const auto &ref){ return ref.var.name; }},
        m_Detail);
}
//----------------------------------------------------------------------------
unsigned int VarReference::getNumNeurons() const
{
    return std::visit(
        Utils::Overload{
            [](const NGRef &ref) { return ref.group->getNumNeurons(); },
            [](const PSMRef &ref) { return ref.group->getTrgNeuronGroup()->getNumNeurons(); },
            [](const WUPreRef &ref) { return ref.group->getSrcNeuronGroup()->getNumNeurons(); },
            [](const WUPostRef &ref) { return ref.group->getTrgNeuronGroup()->getNumNeurons(); },
            [](const CSRef &ref) { return ref.group->getTrgNeuronGroup()->getNumNeurons(); },
            [](const CURef &ref) { return ref.group->getNumNeurons(); },
            [](const CCUPreRef &ref) { return ref.group->getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); },
            [](const CCUPostRef &ref) { return ref.group->getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons(); },
            [](const InternalRef &ref)
            {
                if(ref.type == InternalRef::Type::OUT_POST) {
                    return ref.group->getTrgNeuronGroup()->getNumNeurons();
                }
                else {
                    return ref.group->getSrcNeuronGroup()->getNumNeurons();
                }
            }},
        m_Detail);
}
//----------------------------------------------------------------------------
bool VarReference::isTargetNeuronGroup(const NeuronGroupInternal *ng) const
{
    return std::visit(
        Utils::Overload{
            [ng](const NGRef &ref){ return (ref.group == ng); },
            [](const auto&) { return false; }},
        m_Detail);
}
//----------------------------------------------------------------------------
NeuronGroup *VarReference::getDelayNeuronGroup() const
{ 
    return std::visit(
        Utils::Overload{
            [](const NGRef &ref)->NeuronGroup* {
                // **NOTE** this will also work fine with dendritic delay
                return (ref.group->isDelayRequired() && ref.group->isVarQueueRequired(ref.var.name)) ? ref.group : nullptr;
            },
            [](const WUPreRef &ref)->NeuronGroup* {
                return (ref.group->getAxonalDelaySteps() > 0) ? ref.group->getSrcNeuronGroup() : nullptr;
            },
            [](const WUPostRef &ref)->NeuronGroup* {
                return (ref.group->getBackPropDelaySteps() > 0 
                        || ref.group->getWUInitialiser().isVarHeterogeneouslyDelayedInSynCode(ref.var.name)) ? ref.group->getTrgNeuronGroup() : nullptr;
            },
            [](const auto&)->NeuronGroup* { return nullptr; }},
        m_Detail);
}
//----------------------------------------------------------------------------
const Runtime::ArrayBase *VarReference::getTargetArray(const Runtime::Runtime &runtime) const
{ 
    return std::visit(
        Utils::Overload{
            [&runtime](const PSMRef &ref) { return runtime.getArray(ref.group->getFusedPSTarget(), ref.var.name); },
            [&runtime](const WUPreRef &ref) { return runtime.getArray(ref.group->getFusedWUPreTarget(), ref.var.name); },
            [&runtime](const WUPostRef &ref) { return runtime.getArray(ref.group->getFusedWUPostTarget(), ref.var.name); },
            [&runtime](const InternalRef &ref) 
            {
                if(ref.type == InternalRef::Type::OUT_POST) {
                    return runtime.getArray(ref.group->getFusedPSTarget(), "outPost");
                }
                else {
                    return runtime.getArray(ref.group->getFusedPreOutputTarget(), "outPre");
                }
            },
            [&runtime](const auto &ref) { return runtime.getArray(*ref.group, ref.var.name); }},
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
    const std::string varName = getVarName();
    const std::string &targetName = getTargetName();
    const std::string &otherVarName = other.getVarName();
    const std::string &otherTargetName = other.getTargetName();

    return (std::tie(varName, targetName) 
            < std::tie(otherVarName, otherTargetName)); 
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(NeuronGroup *ng, const std::string &varName)
{
    const auto *nm = ng->getModel();
    try {
        return VarReference(NGRef{static_cast<NeuronGroupInternal*>(ng), nm->getVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(CurrentSource *cs, const std::string &varName)
{
    const auto *csm = cs->getModel();
    try {
        return VarReference(CSRef{static_cast<CurrentSourceInternal*>(cs), csm->getVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::createVarRef(CustomUpdate *cu, const std::string &varName)
{
    const auto *cum = cu->getModel();
    try {
        return VarReference(CURef{static_cast<CustomUpdateInternal*>(cu), cum->getVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::createPreVarRef(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    const auto *ccum = ccu->getModel();
    try {
        return VarReference(CCUPreRef{static_cast<CustomConnectivityUpdateInternal*>(ccu), ccum->getPreVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::createPostVarRef(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    const auto *ccum = ccu->getModel();
    try {
        return VarReference(CCUPostRef{static_cast<CustomConnectivityUpdateInternal*>(ccu), ccum->getPostVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::createPSMVarRef(SynapseGroup *sg, const std::string &varName)
{
    const auto *psm = sg->getPSInitialiser().getSnippet();
    try {
        return VarReference(PSMRef{static_cast<SynapseGroupInternal*>(sg), psm->getVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPreVarRef(SynapseGroup *sg, const std::string &varName)
{
    const auto *wum = sg->getWUInitialiser().getSnippet();
    try {
        return VarReference(WUPreRef{static_cast<SynapseGroupInternal*>(sg), wum->getPreVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::createWUPostVarRef(SynapseGroup *sg, const std::string &varName)
{
    const auto *wum = sg->getWUInitialiser().getSnippet();
    try {
        return VarReference(WUPostRef{static_cast<SynapseGroupInternal*>(sg), wum->getPostVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//----------------------------------------------------------------------------
VarReference VarReference::createOutPostRef(SynapseGroup *sg)
{
    return VarReference(InternalRef{static_cast<SynapseGroupInternal*>(sg),
                                    InternalRef::Type::OUT_POST});
}
//----------------------------------------------------------------------------
VarReference VarReference::createOutPreRef(SynapseGroup *sg)
{
    return VarReference(InternalRef{static_cast<SynapseGroupInternal*>(sg),
                                    InternalRef::Type::OUT_PRE});
}
//----------------------------------------------------------------------------
const std::string &VarReference::getTargetName() const 
{ 
    return std::visit(
        Utils::Overload{
            [](const PSMRef &ref) { return std::cref(ref.group->getFusedPSTarget().getName()); },
            [](const WUPreRef &ref) { return std::cref(ref.group->getFusedWUPreTarget().getName()); },
            [](const WUPostRef &ref) { return std::cref(ref.group->getFusedWUPostTarget().getName()); },
            [](const InternalRef &ref) 
            {
                if(ref.type == InternalRef::Type::OUT_POST) {
                    return std::cref(ref.group->getFusedPSTarget().getName());
                }
                else {
                    return std::cref(ref.group->getFusedPreOutputTarget().getName());
                }
            },
            [](const auto &ref) { return std::cref(ref.group->getName()); }},
        m_Detail);
}

//----------------------------------------------------------------------------
// WUVarReference
//----------------------------------------------------------------------------
Type::UnresolvedType WUVarReference::getVarType() const
{
    return std::visit(
        [](const auto &ref){ return ref.var.type; },
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
const Runtime::ArrayBase *WUVarReference::getTargetArray(const Runtime::Runtime &runtime) const
{
    return std::visit(
        [&runtime](const auto &ref) { return runtime.getArray(*ref.group, ref.var.name); },
        m_Detail);
}
//----------------------------------------------------------------------------
SynapseGroup *WUVarReference::getSynapseGroup() const
{
    return getSynapseGroupInternal();
}
//------------------------------------------------------------------------
/*std::optional<std::string> WUVarReference::getTransposeVarName() const
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
}*/
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
const Runtime::ArrayBase *WUVarReference::getTransposeTargetArray(const Runtime::Runtime &runtime) const
{
    return std::visit(
        Utils::Overload{
            [this, &runtime](const WURef &ref)->Runtime::ArrayBase*
            { 
                if(ref.transposeVar) {
                    return runtime.getArray(*getTransposeSynapseGroup(), ref.transposeVar->name);
                }
                else {
                    return nullptr;
                }
            },
            [](const auto&)->Runtime::ArrayBase*{ return nullptr; }},
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
    const auto *wum = sg->getWUInitialiser().getSnippet();
    auto *sgInternal = static_cast<SynapseGroupInternal*>(sg);
    
    const auto var = wum->getVar(varName);
    if(var) {
        if(transposeSG) {
            const auto *transposeWUM = transposeSG->getWUInitialiser().getSnippet();
            try {
                return WUVarReference(WURef{sgInternal, static_cast<SynapseGroupInternal*>(transposeSG),
                                            var.value(), transposeWUM->getVar(transposeVarName).value()});
            }
            catch(std::bad_optional_access&) {
                throw std::runtime_error("Transpose variable '" + transposeVarName + "' not found");
            }
        }
        else {
            return WUVarReference(WURef{static_cast<SynapseGroupInternal*>(sg), nullptr, var.value(), std::nullopt});
        }
    }
    else {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
}
//------------------------------------------------------------------------
WUVarReference WUVarReference::createWUVarReference(CustomUpdateWU *cu, const std::string &varName)
{
    const auto *cum = cu->getModel();
    try {
        return WUVarReference(CURef{static_cast<CustomUpdateWUInternal*>(cu), cum->getVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Transpose variable '" + varName + "' not found");
    }
}
//------------------------------------------------------------------------
WUVarReference WUVarReference::createWUVarReference(CustomConnectivityUpdate *ccu, const std::string &varName)
{
    const auto *ccum = ccu->getModel();
    try {
        return WUVarReference(CCURef{static_cast<CustomConnectivityUpdateInternal*>(ccu), ccum->getVar(varName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Variable '" + varName + "' not found");
    }
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
const std::string &WUVarReference::getVarName() const
{
    return std::visit(
        [](const auto &ref){ return std::cref(ref.var.name); },
        m_Detail);
}
//----------------------------------------------------------------------------
const std::string &WUVarReference::getTargetName() const
{
    return std::visit(
        [](const auto &ref) { return std::cref(ref.group->getName()); },
        m_Detail);
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

//----------------------------------------------------------------------------
// EGPReference
//----------------------------------------------------------------------------
const Models::Base::EGP &EGPReference::getEGP() const
{
    return std::visit(
        [](const auto &ref) { return std::cref(ref.egp); },
        m_Detail);
}
 //----------------------------------------------------------------------------
const Runtime::ArrayBase *EGPReference::getTargetArray(const Runtime::Runtime &runtime) const
{
    return std::visit(
        [&runtime](const auto &ref) { return runtime.getArray(*ref.group, ref.egp.name); },
        m_Detail);
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(NeuronGroup *ng, const std::string &egpName)
{
    const auto *nm = ng->getModel();
    try {
        return EGPReference(NGRef{ng, nm->getExtraGlobalParam(egpName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Extra global parameter '" + egpName + "' not found");
    }
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(CurrentSource *cs, const std::string &egpName)
{
    const auto *cm = cs->getModel();
    try {
        return EGPReference(CSRef{cs, cm->getExtraGlobalParam(egpName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Extra global parameter '" + egpName + "' not found");
    }
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(CustomUpdate *cu, const std::string &egpName)
{
    const auto *cm = cu->getModel();
    try {
        return EGPReference(CURef{cu, cm->getExtraGlobalParam(egpName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Extra global parameter '" + egpName + "' not found");
    }
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(CustomUpdateWU *cu, const std::string &egpName)
{
    const auto *cm = cu->getModel();
    try {
        return EGPReference(CUWURef{cu, cm->getExtraGlobalParam(egpName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Extra global parameter '" + egpName + "' not found");
    }
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createEGPRef(CustomConnectivityUpdate *ccu, const std::string &egpName)
{
    const auto *ccm = ccu->getModel();
    try {
        return EGPReference(CCURef{ccu, ccm->getExtraGlobalParam(egpName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Extra global parameter '" + egpName + "' not found");
    }
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createPSMEGPRef(SynapseGroup *sg, const std::string &egpName)
{
    const auto *psm = sg->getPSInitialiser().getSnippet();
    try {
        return EGPReference(PSMRef{sg, psm->getExtraGlobalParam(egpName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Extra global parameter '" + egpName + "' not found");
    }
}
//----------------------------------------------------------------------------
EGPReference EGPReference::createWUEGPRef(SynapseGroup *sg, const std::string &egpName)
{
    const auto *wum = sg->getWUInitialiser().getSnippet();
    try {
        return EGPReference(WURef{sg, wum->getExtraGlobalParam(egpName).value()});
    }
    catch(std::bad_optional_access&) {
        throw std::runtime_error("Extra global parameter '" + egpName + "' not found");
    }
}
//----------------------------------------------------------------------------
const std::string &EGPReference::getEGPName() const
{
    return std::visit(
        [](const auto &ref){ return std::cref(ref.egp.name); },
        m_Detail);
}
//----------------------------------------------------------------------------
const std::string &EGPReference::getTargetName() const
{
    return std::visit(
        [](const auto &ref) { return std::cref(ref.group->getName()); },
        m_Detail);
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
void checkLocalVarReferences(const std::map<std::string, VarReference> &varRefs, const Base::VarRefVec &modelVarRefs,
                             const NeuronGroupInternal *ng, const std::string &targetErrorDescription)
{
    // Loop through all variable references
    for(const auto &modelVarRef : modelVarRefs) {
        const auto &varRef = varRefs.at(modelVarRef.name);

        // If (neuron) variable being targetted doesn't have BATCH or ELEMENT axis,
        // check it's only accessed read-only
        const auto varDims = varRef.getVarDims();
        if((!(varDims & VarAccessDim::BATCH) || !(varDims & VarAccessDim::ELEMENT))
            && (modelVarRef.access != VarAccessMode::READ_ONLY))
        {
            throw std::runtime_error("Variable references to SHARED_NEURON or SHARED neuron variables cannot be read-write.");
        }

        // If variable reference target doesn't belong to neuron group, give error
        if(!varRef.isTargetNeuronGroup(ng)) {
            throw std::runtime_error(targetErrorDescription);
        }
    }
}
//----------------------------------------------------------------------------
void checkEGPReferenceTypes(const std::map<std::string, EGPReference> &egpRefs,
                            const Base::EGPRefVec &modelEGPRefs)
{
    // Loop through all extra global parameter references
    for (const auto &modelEGPRef : modelEGPRefs) {
        const auto egpRef = egpRefs.at(modelEGPRef.name);

        // Check types of extra global parameter references against those specified in model
        // **THINK** this is rather conservative but I think not allowing "scalar" and whatever happens to be scalar type is ok
        if (egpRef.getEGP().type != modelEGPRef.type) {
            throw std::runtime_error("Incompatible type for extra global parameter reference '" + modelEGPRef.name + "'");
        }
    }
}
}   // namespace GeNN::Models
