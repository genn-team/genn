#include "code_generator/customUpdateGroupMerged.h"

// Standard C++ includes
#include <sstream>

// GeNN code generator includes
#include "code_generator/environment.h"
#include "code_generator/groupMergedTypeEnvironment.h"
#include "code_generator/modelSpecMerged.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/prettyPrinter.h"
#include "transpiler/scanner.h"
#include "transpiler/typeChecker.h"
#include "transpiler/transpilerUtils.h"


using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateGroupMerged::name = "CustomUpdate";
//----------------------------------------------------------------------------
CustomUpdateGroupMerged::CustomUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                 const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   GroupMerged<CustomUpdateInternal>(index, typeContext, groups)
{
    using namespace Type;

    // Create type environment
    TypeChecker::StandardLibraryFunctionEnvironment stdLibraryEnv;
    GroupMergedTypeEnvironment<CustomUpdateGroupMerged> typeEnvironment(*this, &stdLibraryEnv);

    addField<Uint32>("size", [](const auto &c, size_t) { return std::to_string(c.getSize()); });
    
    // If some variables are delayed, add delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        addField(Uint32::getInstance()->getPointerType(), "spkQuePtr", 
                 [&backend](const auto &cg, size_t) 
                 { 
                     return backend.getScalarAddressPrefix() + "spkQuePtr" + cg.getDelayNeuronGroup()->getName(); 
                 });
    }

    // Add heterogeneous custom update model parameters
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    typeEnvironment.defineHeterogeneousParams<CustomUpdateGroupMerged>(
        cm->getParamNames(), "",
        [](const auto &cg) { return cg.getParams(); },
        &CustomUpdateGroupMerged::isParamHeterogeneous);

    // Add heterogeneous weight update model derived parameters
    typeEnvironment.defineHeterogeneousDerivedParams<CustomUpdateGroupMerged>(
        cm->getDerivedParams(), "",
        [](const auto &cg) { return cg.getDerivedParams(); },
        &CustomUpdateGroupMerged::isDerivedParamHeterogeneous);

    // Add variables to struct
    typeEnvironment.defineVars(cm->getVars(), backend.getDeviceVarPrefix());

    // Add variable references to struct
    typeEnvironment.defineVarReferences(cm->getVarRefs(), backend.getDeviceVarPrefix(),
                    [](const auto &cg) { return cg.getVarReferences(); });

     // Add EGPs to struct
     typeEnvironment.defineEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());

     // Scan, parse and type-check update code
     ErrorHandler errorHandler;
     const std::string code = upgradeCodeString(cm->getUpdateCode());
     const auto tokens = Scanner::scanSource(code, errorHandler);
     m_UpdateStatements = Parser::parseBlockItemList(tokens, errorHandler);
     TypeChecker::typeCheck(m_UpdateStatements, typeEnvironment, typeContext, errorHandler);
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

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomUpdateGroupMerged::generateCustomUpdate(const BackendBase &backend, EnvironmentExternal &env) const
{
     // Add parameters, derived parameters and EGPs to environment
    EnvironmentSubstitute envSubs(env);
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    envSubs.addParamValueSubstitution(cm->getParamNames(), getArchetype().getParams(),
                                     [this](const std::string &p) { return isParamHeterogeneous(p); });
    envSubs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(),
                                    [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  });
    envSubs.addVarNameSubstitution(cm->getExtraGlobalParams());

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarAdapter, CustomUpdateInternal> varSubs(
        getArchetype(), envSubs, 
        [this](const Models::VarInit&, VarAccess a)
        {
            return getVarIndex(getVarAccessDuplication(a), "id");
        });
    
    // Create an environment which caches variable references in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarRefAdapter, CustomUpdateInternal> varRefSubs(
        getArchetype(), varSubs, 
        [this](const Models::VarReference &v, VarAccessMode)
        { 
            return getVarRefIndex(v.getDelayNeuronGroup() != nullptr, 
                                  getVarAccessDuplication(v.getVar().access), 
                                  "id");
        });

    // Pretty print previously parsed update statements
    PrettyPrinter::print(getUpdateStatements(), varRefSubs, getTypeContext());
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

    return hash.get_digest();
}
//----------------------------------------------------------------------------
void CustomUpdateWUGroupMergedBase::generateCustomUpdate(const BackendBase &backend, EnvironmentExternal &env) const
{
     // Add parameters, derived parameters and EGPs to environment
    EnvironmentSubstitute envSubs(env);
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    envSubs.addParamValueSubstitution(cm->getParamNames(), getArchetype().getParams(),
                                     [this](const std::string &p) { return isParamHeterogeneous(p); });
    envSubs.addVarValueSubstitution(cm->getDerivedParams(), getArchetype().getDerivedParams(),
                                    [this](const std::string &p) { return isDerivedParamHeterogeneous(p);  });
    envSubs.addVarNameSubstitution(cm->getExtraGlobalParams());

    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarAdapter, CustomUpdateWUInternal> varSubs(
        getArchetype(), envSubs, 
        [this](const Models::VarInit&, VarAccess a)
        {
            return getVarIndex(getVarAccessDuplication(a), "id_syn");
        });
    
    // Create an environment which caches variable references in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateWUVarRefAdapter, CustomUpdateWUInternal> varRefSubs(
        getArchetype(), varSubs, 
        [this](const Models::WUVarReference &v, VarAccessMode)
        { 
            return getVarRefIndex(getVarAccessDuplication(v.getVar().access),
                                  "id_syn");
        });

    // Pretty print previously parsed update statements
    PrettyPrinter::print(getUpdateStatements(), varRefSubs, getTypeContext());
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
CustomUpdateWUGroupMergedBase::CustomUpdateWUGroupMergedBase(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                             const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   GroupMerged<CustomUpdateWUInternal>(index, typeContext, groups)
{
    using namespace Type;

    // Create type environment
    TypeChecker::StandardLibraryFunctionEnvironment stdLibraryEnv;
    GroupMergedTypeEnvironment<CustomUpdateWUGroupMergedBase> typeEnvironment(*this, &stdLibraryEnv);

    // If underlying synapse group has kernel weights
    if (getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixWeight::KERNEL) {
        // Loop through kernel size dimensions
        for (size_t d = 0; d < getArchetype().getSynapseGroup()->getKernelSize().size(); d++) {
            // If this dimension has a heterogeneous size, add it to struct
            if (isKernelSizeHeterogeneous(d)) {
                addField<Uint32>("kernelSize" + std::to_string(d),
                                 [d](const auto &cu, size_t) 
                                 {
                                     return std::to_string(cu.getSynapseGroup()->getKernelSize().at(d));
                                 });
            }
        }
    }
    // Otherwise
    else {
        addField<Uint32>("rowStride",
                         [&backend](const auto &cg, size_t) 
                         { 
                             const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                             return std::to_string(backend.getSynapticMatrixRowStride(*sgInternal)); 
                         });
    
        addField<Uint32>("numSrcNeurons",
                         [](const auto &cg, size_t) 
                         {
                             const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                             return std::to_string(sgInternal->getSrcNeuronGroup()->getNumNeurons()); 
                         });

        addField<Uint32>("numTrgNeurons",
                         [](const auto &cg, size_t)
                         { 
                             const SynapseGroupInternal *sgInternal = static_cast<const SynapseGroupInternal*>(cg.getSynapseGroup());
                             return std::to_string(sgInternal->getTrgNeuronGroup()->getNumNeurons()); 
                         });

        // If synapse group has sparse connectivity
        if(getArchetype().getSynapseGroup()->getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            addField(getArchetype().getSynapseGroup()->getSparseIndType()->getPointerType(), "ind", 
                     [&backend](const auto &cg, size_t) 
                     { 
                         return backend.getDeviceVarPrefix() + "ind" + cg.getSynapseGroup()->getName(); 
                     });

            addField(Uint32::getInstance()->getPointerType(), "rowLength",
                     [&backend](const auto &cg, size_t) 
                     { 
                         return backend.getDeviceVarPrefix() + "rowLength" + cg.getSynapseGroup()->getName(); 
                     });
        }
    }

    // Add heterogeneous custom update model parameters
    const CustomUpdateModels::Base *cm = getArchetype().getCustomUpdateModel();
    typeEnvironment.defineHeterogeneousParams<CustomUpdateWUGroupMerged>(
        cm->getParamNames(), "",
        [](const auto &cg) { return cg.getParams(); },
        &CustomUpdateWUGroupMergedBase::isParamHeterogeneous);

    // Add heterogeneous weight update model derived parameters
    typeEnvironment.defineHeterogeneousDerivedParams<CustomUpdateWUGroupMerged>(
        cm->getDerivedParams(), "",
        [](const auto &cg) { return cg.getDerivedParams(); },
        &CustomUpdateWUGroupMergedBase::isDerivedParamHeterogeneous);

    // Add variables to struct
    typeEnvironment.defineVars(cm->getVars(), backend.getDeviceVarPrefix());

    // Add variable references to struct
    const auto varRefs = cm->getVarRefs();
    typeEnvironment.defineVarReferences(varRefs, backend.getDeviceVarPrefix(),
                                        [](const auto &cg) { return cg.getVarReferences(); });

     // Loop through variables
    for(const auto &v : varRefs) {
        // If variable has a transpose 
        if(getArchetype().getVarReferences().at(v.name).getTransposeSynapseGroup() != nullptr) {
            // Add field with transpose suffix, pointing to transpose var
            addField(v.type->getPointerType(), v.name + "Transpose",
                     [&backend, v](const auto &g, size_t)
                     {
                         const auto varRef = g.getVarReferences().at(v.name);
                         return backend.getDeviceVarPrefix() + varRef.getTransposeVar().name + varRef.getTransposeTargetName();
                     });
            }
    }
    // Add EGPs to struct
    typeEnvironment.defineEGPs(cm->getExtraGlobalParams(), backend.getDeviceVarPrefix());

    // Scan, parse and type-check update code
    ErrorHandler errorHandler;
    const std::string code = upgradeCodeString(cm->getUpdateCode());
    const auto tokens = Scanner::scanSource(code, errorHandler);
    m_UpdateStatements = Parser::parseBlockItemList(tokens, errorHandler);
    TypeChecker::typeCheck(m_UpdateStatements, typeEnvironment, typeContext, errorHandler);
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
// CustomUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateHostReductionGroupMerged::name = "CustomUpdateHostReduction";
//----------------------------------------------------------------------------
CustomUpdateHostReductionGroupMerged::CustomUpdateHostReductionGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                           const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups)
:   CustomUpdateHostReductionGroupMergedBase<CustomUpdateInternal>(index, typeContext, backend, groups)
{
    using namespace Type;

    addField<Uint32>("size",
                     [](const auto &c, size_t) { return std::to_string(c.getSize()); });

    // If some variables are delayed, add delay pointer
    // **NOTE** this is HOST delay pointer
    if(getArchetype().getDelayNeuronGroup() != nullptr) {
        addField(Uint32::getInstance()->getPointerType(), "spkQuePtr", 
                 [](const auto &cg, size_t) 
                 { 
                     return "spkQuePtr" + cg.getDelayNeuronGroup()->getName(); 
                 });
    }
}

// ----------------------------------------------------------------------------
// CustomWUUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
const std::string CustomWUUpdateHostReductionGroupMerged::name = "CustomWUUpdateHostReduction";
//----------------------------------------------------------------------------
CustomWUUpdateHostReductionGroupMerged::CustomWUUpdateHostReductionGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                                               const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
:   CustomUpdateHostReductionGroupMergedBase<CustomUpdateWUInternal>(index, typeContext, backend, groups)
{
    using namespace Type;

    addField<Uint32>("size",
                     [&backend](const auto &cg, size_t) 
                     {
                         return std::to_string(cg.getSynapseGroup()->getMaxConnections() * (size_t)cg.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons()); 
                     });
}
