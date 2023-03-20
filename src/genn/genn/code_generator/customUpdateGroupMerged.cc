#include "code_generator/customUpdateGroupMerged.h"

// Standard C++ includes
#include <sstream>

// GeNN code generator includes
#include "code_generator/environment.h"
#include "code_generator/modelSpecMerged.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/prettyPrinter.h"
#include "transpiler/scanner.h"
#include "transpiler/typeChecker.h"


using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateGroupMerged::name = "CustomUpdate";
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateGroupMerged::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with each group's custom update size
    updateHash([](const auto &cg) { return cg.getSize(); }, hash);

    // Update hash with each group's parameters, derived parameters and variable references
    updateHash([](const auto &cg) { return cg.getParams(); }, hash);
    updateHash([](const auto &cg) { return cg.getDerivedParams(); }, hash);
    updateHash([](const auto &cg) { return cg.getVarReferences(); }, hash);
    updateHash([](const auto &cg) { return cg.getEGPReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomUpdateGroupMerged::generateCustomUpdate(const BackendBase &backend, EnvironmentExternalBase &env)
{
    // Add parameters, derived parameters and EGPs to environment
    EnvironmentGroupMergedField<CustomUpdateGroupMerged> cuEnv(env, *this);

    cuEnv.addField(Type::Uint32.addConst(), "size",
                   Type::Uint32, "size",
                   [](const CustomUpdateInternal &c, size_t) { return std::to_string(c.getSize()); });
    
    // If some variables are delayed, add delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        cuEnv.addField(Type::Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr",
                       [&backend](const auto &g, size_t) { return backend.getScalarAddressPrefix() + "spkQuePtr" + g.getDelayNeuronGroup()->getName(); });
    }

    // Substitute parameter and derived parameter names
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    cuEnv.addParams(cm->getParamNames(), "", &CustomUpdateInternal::getParams, &CustomUpdateGroupMerged::isParamHeterogeneous);
    cuEnv.addDerivedParams(cm->getDerivedParams(), "", &CustomUpdateInternal::getDerivedParams, &CustomUpdateGroupMerged::isDerivedParamHeterogeneous);
    cuEnv.addExtraGlobalParams(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());
    cuEnv.addExtraGlobalParamRefs(cm->getExtraGlobalParamRefs(), backend.getDeviceVarPrefix());

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarAdapter, CustomUpdateGroupMerged> varEnv(
        *this, *this, getTypeContext(), cuEnv, backend.getDeviceVarPrefix(), "", "l",
        [this, &cuEnv](const std::string&, VarAccessDuplication d)
        {
            return getVarIndex(d, cuEnv["id"]);
        });
    
    // Create an environment which caches variable references in local variables if they are accessed
    EnvironmentLocalVarRefCache<CustomUpdateVarRefAdapter, CustomUpdateGroupMerged> varRefEnv(
        *this, *this, getTypeContext(), varEnv, backend.getDeviceVarPrefix(), "", "l", 
        [this, &varEnv](const std::string&, const Models::VarReference &v)
        { 
            return getVarRefIndex(v.getDelayNeuronGroup() != nullptr, 
                                  getVarAccessDuplication(v.getVar().access), 
                                  varEnv["id"]);
        });

    Transpiler::ErrorHandler errorHandler("Custom update '" + getArchetype().getName() + "' update code");
    prettyPrintExpression(getArchetype().getUpdateCodeTokens(), getTypeContext(), varRefEnv, errorHandler);
}
//----------------------------------------------------------------------------
std::string CustomUpdateGroupMerged::getVarIndex(VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
        return getArchetype().isBatched() ? "batch" : "0";
    }
    else if (varDuplication == VarAccessDuplication::SHARED || !getArchetype().isBatched()) {
        assert(!index.empty());
        return index;
    }
    else {
        assert(!index.empty());
        return "batchOffset + " + index;
    }
}
//----------------------------------------------------------------------------
std::string CustomUpdateGroupMerged::getVarRefIndex(bool delay, VarAccessDuplication varDuplication, const std::string &index) const
{
    // If delayed, variable is shared, the batch size is one or this custom update isn't batched, batch delay offset isn't required
    if(delay) {
        if (varDuplication == VarAccessDuplication::SHARED_NEURON) {
            return getArchetype().isBatched() ? "batchDelaySlot" : "delaySlot";
        }
        else if (varDuplication == VarAccessDuplication::SHARED || !getArchetype().isBatched()) {
            assert(!index.empty());
            return "delayOffset + " + index;
        }
        else {
            assert(!index.empty());
            return "batchDelayOffset + " + index;
        }
    }
    else {
        return getVarIndex(varDuplication, index);
    }    
}
//----------------------------------------------------------------------------
bool CustomUpdateGroupMerged::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const auto &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------    
bool CustomUpdateGroupMerged::isDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const auto &cg) { return cg.getDerivedParams(); });
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateWUGroupMergedBase
//----------------------------------------------------------------------------
bool CustomUpdateWUGroupMergedBase::isParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const CustomUpdateWUInternal &cg) { return cg.getParams(); });
}
//----------------------------------------------------------------------------
bool CustomUpdateWUGroupMergedBase::isDerivedParamHeterogeneous(const std::string &paramName) const
{
    return isParamValueHeterogeneous(paramName, [](const CustomUpdateWUInternal &cg) { return cg.getDerivedParams(); });
}
//----------------------------------------------------------------------------
boost::uuids::detail::sha1::digest_type CustomUpdateWUGroupMergedBase::getHashDigest() const
{
    boost::uuids::detail::sha1 hash;

    // Update hash with archetype's hash digest
    Utils::updateHash(getArchetype().getHashDigest(), hash);

    // Update hash with sizes of pre and postsynaptic neuron groups
    updateHash([](const auto &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getSrcNeuronGroup()->getNumNeurons();
               }, hash);

    updateHash([](const auto &cg) 
               {
                   return static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup())->getTrgNeuronGroup()->getNumNeurons();
               }, hash);

    // Update hash with each group's parameters, derived parameters and variable referneces
    updateHash([](const auto &cg) { return cg.getParams(); }, hash);
    updateHash([](const auto &cg) { return cg.getDerivedParams(); }, hash);
    updateHash([](const auto &cg) { return cg.getVarReferences(); }, hash);
    updateHash([](const auto &cg) { return cg.getEGPReferences(); }, hash);

    return hash.get_digest();
}
//----------------------------------------------------------------------------
std::string CustomUpdateWUGroupMergedBase::getVarIndex(VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    return ((varDuplication == VarAccessDuplication::SHARED || !getArchetype().isBatched()) ? "" : "batchOffset + ") + index;
}
//----------------------------------------------------------------------------
std::string CustomUpdateWUGroupMergedBase::getVarRefIndex(VarAccessDuplication varDuplication, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    return ((varDuplication == VarAccessDuplication::SHARED || !getArchetype().isBatched()) ? "" : "batchOffset + ") + index;
}
//----------------------------------------------------------------------------
void CustomUpdateWUGroupMergedBase::generateCustomUpdateBase(const BackendBase &backend, EnvironmentExternalBase &env)
{
    // Add parameters, derived parameters and EGPs to environment
    EnvironmentGroupMergedField<CustomUpdateWUGroupMergedBase> cuEnv(env, *this);

    // If underlying synapse group has kernel weights
    if (getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL) {
        // Loop through kernel size dimensions
        for (size_t d = 0; d < getArchetype().getSynapseGroup()->getKernelSize().size(); d++) {
            // If this dimension has a heterogeneous size, add it to struct
            if (isKernelSizeHeterogeneous(*this, d)) {
                cuEnv.addField(Type::Uint32, "_kernel_size_" + std::to_string(d), "kernelSize" + std::to_string(d),
                               [d](const auto &cu, size_t) 
                               {
                                   return std::to_string(cu.getSynapseGroup()->getKernelSize().at(d));
                               });
            }
        }
    }
    // Otherwise
    else {
        cuEnv.addField(Type::Uint32, "_row_stride", "rowStride", 
                       [&backend](const auto &cg, size_t) 
                       {
                           const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                           return std::to_string(backend.getSynapticMatrixRowStride(*sgInternal)); 
                       });
    
        cuEnv.addField(Type::Uint32, "num_pre", "numSrcNeurons",
                       [](const auto &cg, size_t) 
                       {
                           const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                           return std::to_string(sgInternal->getSrcNeuronGroup()->getNumNeurons()); 
                       });

        cuEnv.addField(Type::Uint32, "num_post", "numTrgNeurons",
                       [](const auto &cg, size_t)
                       { 
                           const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                           return std::to_string(sgInternal->getTrgNeuronGroup()->getNumNeurons()); 
                       });

        // If synapse group has sparse connectivity
        if(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            cuEnv.addField(getArchetype().getSynapseGroup()->getSparseIndType().createPointer(), "_ind", "ind", 
                           [&backend](const auto &cg, size_t) 
                           { 
                               return backend.getDeviceVarPrefix() + "ind" + cg.getSynapseGroup()->getName(); 
                           });

            cuEnv.addField(Type::Uint32.createPointer(), "_row_length", "rowLength",
                           [&backend](const auto &cg, size_t) 
                           { 
                               return backend.getDeviceVarPrefix() + "rowLength" + cg.getSynapseGroup()->getName(); 
                           });
        }
    }

     // Substitute parameter and derived parameter names
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    cuEnv.addParams(cm->getParamNames(), "", &CustomUpdateInternal::getParams, &CustomUpdateWUGroupMergedBase::isParamHeterogeneous);
    cuEnv.addDerivedParams(cm->getDerivedParams(), "", &CustomUpdateInternal::getDerivedParams, &CustomUpdateWUGroupMergedBase::isDerivedParamHeterogeneous);
    cuEnv.addExtraGlobalParams(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());
    cuEnv.addExtraGlobalParamRefs(cm->getExtraGlobalParamRefs(), backend.getDeviceVarPrefix());

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarAdapter, CustomUpdateWUGroupMergedBase> varEnv(
        *this, *this, getTypeContext(), cuEnv, backend.getDeviceVarPrefix(), "", "l",
        [this, &cuEnv](const std::string&, VarAccessDuplication d)
        {
            return getVarIndex(d, cuEnv["id_syn"]);
        });
    
    // Create an environment which caches variable references in local variables if they are accessed
    EnvironmentLocalVarRefCache<CustomUpdateWUVarRefAdapter, CustomUpdateWUGroupMergedBase> varRefEnv(
        *this, *this, getTypeContext(), varEnv, backend.getDeviceVarPrefix(), "", "l",
        [this, &varEnv](const std::string&, const Models::WUVarReference &v)
        { 
            return getVarRefIndex(getVarAccessDuplication(v.getVar().access), 
                                  varEnv["id_syn"]);
        });

    Transpiler::ErrorHandler errorHandler("Custom update '" + getArchetype().getName() + "' update code");
    prettyPrintExpression(getArchetype().getUpdateCodeTokens(), getTypeContext(), varRefEnv, errorHandler);
}
// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateWUGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateWUGroupMerged::name = "CustomUpdateWU";

//----------------------------------------------------------------------------
// CustomUpdateTransposeWUGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateTransposeWUGroupMerged::name = "CustomUpdateTransposeWU";
// ----------------------------------------------------------------------------
void CustomUpdateTransposeWUGroupMerged::generateCustomUpdate(const BackendBase &backend, EnvironmentExternalBase &env)
{
    // Add parameters, derived parameters and EGPs to environment
    EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> cuEnv(env, *this);

    // Loop through variable references
    const auto varRefs = getArchetype().getCustomUpdateModel()->getVarRefs();
    for(const auto &v : varRefs) {
        const auto fieldType = v.type.resolve(getTypeContext()).createPointer();

        // If variable has a transpose, add field with transpose suffix, pointing to transpose var
        if(getArchetype().getVarReferences().at(v.name).getTransposeSynapseGroup() != nullptr) {
            cuEnv.addField(fieldType, v.name + "_transpose", v.name + "Transpose",
                           [&backend, v](const auto &g, size_t)
                           {
                               const auto varRef = g.getVarReferences().at(v.name);
                               return backend.getDeviceVarPrefix() + varRef.getTransposeVar().name + varRef.getTransposeTargetName();
                           });
        }
    }

    generateCustomUpdateBase(backend, cuEnv);
}

// ----------------------------------------------------------------------------
// CustomUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateHostReductionGroupMerged::name = "CustomUpdateHostReduction";
//----------------------------------------------------------------------------
void CustomUpdateHostReductionGroupMerged::generateCustomUpdate(const BackendBase &backend, EnvironmentGroupMergedField<CustomUpdateHostReductionGroupMerged> &env)
{
    env.addField(Type::Uint32, "_size", "size",
                 [](const auto &c, size_t) { return std::to_string(c.getSize()); });
    
    // If some variables are delayed, add delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        env.addField(Type::Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr",
                     [&backend](const auto &g, size_t) { return backend.getScalarAddressPrefix() + "spkQuePtr" + g.getDelayNeuronGroup()->getName(); });
    }

    generateCustomUpdateBase(backend, env);
}
// ----------------------------------------------------------------------------
// CustomWUUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateHostReductionGroupMerged::name = "CustomWUUpdateHostReduction";
//----------------------------------------------------------------------------
void CustomWUUpdateHostReductionGroupMerged::generateCustomUpdate(const BackendBase &backend, EnvironmentGroupMergedField<CustomWUUpdateHostReductionGroupMerged> &env)
{
    env.addField(Type::Uint32, "_size", "size",
                 [](const auto &c, size_t) 
                 { 
                     return std::to_string(c.getSynapseGroup()->getMaxConnections() * (size_t)c.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); 
                 });

    generateCustomUpdateBase(backend, env);
}
