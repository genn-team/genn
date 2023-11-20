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
void CustomUpdateGroupMerged::generateCustomUpdate(EnvironmentExternalBase &env, unsigned int batchSize,
                                                   BackendBase::GroupHandlerEnv<CustomUpdateGroupMerged> genPostamble)
{
    // Add parameters, derived parameters and EGPs to environment
    EnvironmentGroupMergedField<CustomUpdateGroupMerged> cuEnv(env, *this);

    // Substitute parameter and derived parameter names
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    cuEnv.addParams(cm->getParams(), "", &CustomUpdateInternal::getParams, 
                    &CustomUpdateGroupMerged::isParamHeterogeneous,
                    &CustomUpdateInternal::isParamDynamic);
    cuEnv.addDerivedParams(cm->getDerivedParams(), "", &CustomUpdateInternal::getDerivedParams, &CustomUpdateGroupMerged::isDerivedParamHeterogeneous);
    cuEnv.addExtraGlobalParams(cm->getExtraGlobalParams());
    cuEnv.addExtraGlobalParamRefs(cm->getExtraGlobalParamRefs());

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarAdapter, CustomUpdateGroupMerged> varEnv(
        *this, *this, getTypeContext(), cuEnv, "", "l", false,
        [this, batchSize, &cuEnv](const std::string&, CustomUpdateVarAccess d)
        {
            return getVarIndex(batchSize, getVarAccessDim(d, getArchetype().getDims()), "$(id)");
        });
    
    // Create an environment which caches variable references in local variables if they are accessed
    EnvironmentLocalVarRefCache<CustomUpdateVarRefAdapter, CustomUpdateGroupMerged> varRefEnv(
        *this, *this, getTypeContext(), varEnv, "", "l", false,
        [this, batchSize, &varEnv](const std::string&, const Models::VarReference &v)
        { 
            return getVarRefIndex(v.getDelayNeuronGroup() != nullptr, batchSize,
                                  v.getVarDims(), "$(id)");
        });

    Transpiler::ErrorHandler errorHandler("Custom update '" + getArchetype().getName() + "' update code");
    prettyPrintStatements(getArchetype().getUpdateCodeTokens(), getTypeContext(), varRefEnv, errorHandler);

    // Generate postamble for e.g. reduction logic
    genPostamble(varRefEnv, *this);
}
//----------------------------------------------------------------------------
std::string CustomUpdateGroupMerged::getVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
    if (!(varDims & VarAccessDim::ELEMENT)) {
        return batched ? "$(batch)" : "0";
    }
    else if (batched) {
        assert(!index.empty());
        return "$(_batch_offset) + " + index;
    }
    else {
        assert(!index.empty());
        return index;
    }
}
//----------------------------------------------------------------------------
std::string CustomUpdateGroupMerged::getVarRefIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
{
    // If delayed, variable is shared, the batch size is one or this custom update isn't batched, batch delay offset isn't required
    if(delay) {
        const bool batched = ((varDims & VarAccessDim::BATCH) && batchSize > 1);
        if (!(varDims & VarAccessDim::ELEMENT)) {
            return batched ? "$(_batch_delay_slot)" : "$(_delay_slot)";
        }
        else if (batched) {
            assert(!index.empty());
            return "$(_batch_delay_offset) + " + index;
        }
        
        else {
            assert(!index.empty());
            return "$(_delay_offset) + " + index;
        }
    }
    else {
        return getVarIndex(batchSize, varDims, index);
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
void CustomUpdateWUGroupMergedBase::generateCustomUpdate(EnvironmentExternalBase &env, unsigned int batchSize,
                                                         BackendBase::GroupHandlerEnv<CustomUpdateWUGroupMergedBase> genPostamble)
{
    // Add parameters, derived parameters and EGPs to environment
    EnvironmentGroupMergedField<CustomUpdateWUGroupMergedBase> cuEnv(env, *this);

    // Substitute parameter and derived parameter names
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    cuEnv.addParams(cm->getParams(), "", &CustomUpdateWUInternal::getParams, 
                    &CustomUpdateWUGroupMergedBase::isParamHeterogeneous,
                    &CustomUpdateWUInternal::isParamDynamic);
    cuEnv.addDerivedParams(cm->getDerivedParams(), "", &CustomUpdateWUInternal::getDerivedParams, &CustomUpdateWUGroupMergedBase::isDerivedParamHeterogeneous);
    cuEnv.addExtraGlobalParams(cm->getExtraGlobalParams());
    cuEnv.addExtraGlobalParamRefs(cm->getExtraGlobalParamRefs());

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarAdapter, CustomUpdateWUGroupMergedBase> varEnv(
        *this, *this, getTypeContext(), cuEnv, "", "l", false,
        [this, batchSize, &cuEnv](const std::string&, CustomUpdateVarAccess d)
        {
            return getVarIndex(batchSize, getVarAccessDim(d, getArchetype().getDims()), "$(id_syn)");
        });
    
    // Create an environment which caches variable references in local variables if they are accessed
    EnvironmentLocalVarRefCache<CustomUpdateWUVarRefAdapter, CustomUpdateWUGroupMergedBase> varRefEnv(
        *this, *this, getTypeContext(), varEnv, "", "l", false,
        [this, batchSize, &varEnv](const std::string&, const Models::WUVarReference &v)
        {
            return getVarRefIndex(batchSize, v.getVarDims(), "$(id_syn)");
        });

    Transpiler::ErrorHandler errorHandler("Custom update '" + getArchetype().getName() + "' update code");
    prettyPrintStatements(getArchetype().getUpdateCodeTokens(), getTypeContext(), varRefEnv, errorHandler);

    // Generate postamble for e.g. reduction or transpose logic
    genPostamble(varRefEnv, *this);
}
//----------------------------------------------------------------------------
std::string CustomUpdateWUGroupMergedBase::getVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    return (((varDims & VarAccessDim::BATCH) && batchSize > 1) ? "$(_batch_offset) + " : "") + index;
}
//----------------------------------------------------------------------------
std::string CustomUpdateWUGroupMergedBase::getVarRefIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const
{
    // **YUCK** there's a lot of duplication in these methods - do they belong elsewhere?
    
    return (((varDims & VarAccessDim::BATCH) && batchSize > 1) ? "$(_batch_offset) + " : "") + index;
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
std::string CustomUpdateTransposeWUGroupMerged::addTransposeField(EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> &env)
{
    // Loop through variable references
    const auto varRefs = getArchetype().getCustomUpdateModel()->getVarRefs();
    for(const auto &v : varRefs) {
        // If variable has a transpose, add field with transpose suffix, pointing to transpose var
        if(getArchetype().getVarReferences().at(v.name).getTransposeSynapseGroup() != nullptr) {
            const auto fieldType = v.type.resolve(getTypeContext()).createPointer();
            env.addField(fieldType, v.name + "_transpose", v.name + "Transpose",
                         [v](const auto &runtime, const auto &g, size_t)
                         {
                             return g.getVarReferences().at(v.name).getTransposeTargetArray(runtime);
                         });

            // Return name of transpose variable
            return v.name;
        }
    }
    throw std::runtime_error("No transpose variable found");
}

// ----------------------------------------------------------------------------
// CustomUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateHostReductionGroupMerged::name = "CustomUpdateHostReduction";
//----------------------------------------------------------------------------
void CustomUpdateHostReductionGroupMerged::generateCustomUpdate(EnvironmentGroupMergedField<CustomUpdateHostReductionGroupMerged> &env)
{
    env.addField(Type::Uint32, "_size", "size",
                 [](const auto &, const auto &c, size_t) { return c.getSize(); });
    
    // If some variables are delayed, add delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        env.addField(Type::Uint32.createPointer(), "_spk_que_ptr", "spkQuePtr",
                     [](const auto &runtime, const auto &g, size_t) { return runtime.getArray(*g.getDelayNeuronGroup(), "spkQuePtr"); });
    }

    generateCustomUpdateBase(env);
}
// ----------------------------------------------------------------------------
// CustomWUUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateHostReductionGroupMerged::name = "CustomWUUpdateHostReduction";
//----------------------------------------------------------------------------
void CustomWUUpdateHostReductionGroupMerged::generateCustomUpdate(EnvironmentGroupMergedField<CustomWUUpdateHostReductionGroupMerged> &env)
{
    env.addField(Type::Uint32, "_size", "size",
                 [](const auto &, const auto &c, size_t) 
                 { 
                     return c.getSynapseGroup()->getMaxConnections() * (size_t)c.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(); 
                 });

    generateCustomUpdateBase(env);
}
