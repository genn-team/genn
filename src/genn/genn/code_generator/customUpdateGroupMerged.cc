#include "code_generator/customUpdateGroupMerged.h"

// Standard C++ includes
#include <sstream>

// GeNN code generator includes
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


//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class EnvironmentExternal : public PrettyPrinter::EnvironmentBase
{
public:
    EnvironmentExternal(PrettyPrinter::EnvironmentBase &enclosing)
    :   m_Context(enclosing)
    {
    }
    
    EnvironmentExternal(CodeStream &os)
    :   m_Context(os)
    {
    }
    
    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string define(const Token&)
    {
        throw std::runtime_error("Cannot declare variable in external environment");
    }
    
protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    auto &getContext() const{ return m_Context; }
    
    CodeStream &getContextStream() const
    {
        return std::visit(
            Transpiler::Utils::Overload{
                [](std::reference_wrapper<PrettyPrinter::EnvironmentBase> enclosing)->CodeStream& { return enclosing.get().getStream(); },
                [](std::reference_wrapper<CodeStream> os)->CodeStream& { return os.get(); }},
            getContext());
    }
    
    std::string getContextName(const Token &name) const
    {
        return std::visit(
            Transpiler::Utils::Overload{
                [&name](std::reference_wrapper<PrettyPrinter::EnvironmentBase> enclosing)->std::string { return enclosing.get().getName(name); },
                [&name](std::reference_wrapper<CodeStream>)->std::string { throw std::runtime_error("Variable '" + name.lexeme + "' undefined"); }},
            getContext());
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::variant<std::reference_wrapper<PrettyPrinter::EnvironmentBase>, std::reference_wrapper<CodeStream>> m_Context;
};

//! Standard pretty printing environment simply allowing substitutions to be implemented
class EnvironmentSubstitute : public EnvironmentExternal
{
public:
    EnvironmentSubstitute(PrettyPrinter::EnvironmentBase &enclosing) : EnvironmentExternal(enclosing){}
    EnvironmentSubstitute(CodeStream &os) : EnvironmentExternal(os){}
    
    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const Token &name) final
    {
        // If there isn't a substitution for this name, try and get name from context
        auto sub = m_VarSubstitutions.find(name.lexeme);
        if(sub == m_VarSubstitutions.end()) {
            return getContextName(name);
        }
        // Otherwise, return substitution
        else {
            return sub->second;
        }
    }
    
    virtual CodeStream &getStream() final
    {
        return getContextStream();
    }
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addSubstitution(const std::string &source, const std::string &destination)
    {
        if(!m_VarSubstitutions.emplace(source, destination).second) {
            throw std::runtime_error("Redeclaration of substitution '" + source + "'");
        }
    }
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::unordered_map<std::string, std::string> m_VarSubstitutions;
};

//! Pretty printing environment which caches used variables in local variables
template<typename A, typename G>
class EnvironmentLocalVarCache : public EnvironmentExternal
{
    //! Type of a single definition
    typedef typename std::invoke_result_t<decltype(&A::getDefs), A>::value_type DefType;
    
    //! Type of a single initialiser
    typedef typename std::remove_reference_t<std::invoke_result_t<decltype(&A::getInitialisers), A>>::mapped_type InitialiserType;
    
    //! Function used to provide index strings based on initialiser and access type
    typedef std::function<std::string(InitialiserType, decltype(DefType::access))> GetIndexFn;    

public:
    EnvironmentLocalVarCache(const G &group, PrettyPrinter::EnvironmentBase &enclosing, GetIndexFn getIndex, const std::string &localPrefix = "l")
    :   EnvironmentExternal(enclosing), m_Group(group), m_Contents(m_ContentsStream), m_LocalPrefix(localPrefix), m_GetIndex(getIndex)
    {
        // Add name of each definition to map, initially with value set to value
        const auto defs = A(m_Group).getDefs();
        std::transform(defs.cbegin(), defs.cend(), std::inserter(m_VariablesReferenced, m_VariablesReferenced.end()),
                       [](const auto &v){ return std::make_pair(v.name, false); });
    }
    
    EnvironmentLocalVarCache(const G &group, CodeStream &os, GetIndexFn getIndex, const std::string &localPrefix = "l")
    :   EnvironmentExternal(os), m_Group(group), m_Contents(m_ContentsStream), m_LocalPrefix(localPrefix), m_GetIndex(getIndex)
    {
        // Add name of each definition to map, initially with value set to value
        const auto defs = A(m_Group).getDefs();
        std::transform(defs.cbegin(), defs.cend(), std::inserter(m_VariablesReferenced, m_VariablesReferenced.end()),
                       [](const auto &v){ return std::make_pair(v.name, false); });
    }
    
    ~EnvironmentLocalVarCache()
    {
        A adapter(m_Group);
        
        // Copy definitions which have been referenced into new vector
        const auto defs = adapter.getDefs();
        std::remove_const_t<decltype(defs)> referencedVars;
        std::copy_if(defs.cbegin(), defs.cend(), std::back_inserter(referencedVars),
                     [this](const auto &v){ return m_VariablesReferenced.at(v.name); });
        
        // Loop through referenced variables
        const auto &initialisers = adapter.getInitialisers();
        for(const auto &v : referencedVars) {
            if(v.access & VarAccessMode::READ_ONLY) {
                getContextStream() << "const ";
            }
            getContextStream() << v.type->getName() << " " << m_LocalPrefix << v.name;
            
            // If this isn't a reduction, read value from memory
            // **NOTE** by not initialising these variables for reductions, 
            // compilers SHOULD emit a warning if user code doesn't set it to something
            if(!(v.access & VarAccessModeAttribute::REDUCE)) {
                getContextStream() << " = group->" << v.name << "[" << m_GetIndex(initialisers.at(v.name), v.access) << "]";
            }
            getContextStream() << ";" << std::endl;
        }
        
        // Write contents to context stream
        getContextStream() << m_ContentsStream.str();
        
        // Loop through referenced variables again
        for(const auto &v : referencedVars) {
            // If variables are read-write
            if(v.access & VarAccessMode::READ_WRITE) {
                getContextStream() << "group->" << v.name << "[" << m_GetIndex(initialisers.at(v.name), v.access) << "]";
                getContextStream() << " = " << m_LocalPrefix << v.name << ";" << std::endl;
            }
        }
    }

    //------------------------------------------------------------------------
    // PrettyPrinter::EnvironmentBase virtuals
    //------------------------------------------------------------------------
    virtual std::string getName(const Token &name) final
    {
        // If variable with this name isn't found, try and get name from context
        auto var = m_VariablesReferenced.find(name.lexeme);
        if(var == m_VariablesReferenced.end()) {
            return getContextName(name);
        }
        // Otherwise
        else {
            // Set flag to indicate that variable has been referenced
            var->second = true;
            
            // Add local prefix to variable name
            return m_LocalPrefix + name.lexeme;
        }
    }
    
    virtual CodeStream &getStream() final
    {
        return m_Contents;
    }
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const G &m_Group;
    std::ostringstream m_ContentsStream;
    CodeStream m_Contents;
    const std::string m_LocalPrefix;
    const GetIndexFn m_GetIndex;
    std::unordered_map<std::string, bool> m_VariablesReferenced;
};

template<typename C, typename R>
void genCustomUpdate(CodeStream &os, Substitutions &baseSubs, const C &cg, const std::string &index,
                     R getVarRefIndex)
{
    Substitutions updateSubs(&baseSubs);

    const CustomUpdateModels::Base *cm = cg.getArchetype().getCustomUpdateModel();
    const auto varRefs = cm->getVarRefs();

    // Loop through variables
    for(const auto &v : cm->getVars()) {
        if(v.access & VarAccessMode::READ_ONLY) {
            os << "const ";
        }
        os << v.type->getName() << " l" << v.name;
        
        // If this isn't a reduction, read value from memory
        // **NOTE** by not initialising these variables for reductions, 
        // compilers SHOULD emit a warning if user code doesn't set it to something
        if(!(v.access & VarAccessModeAttribute::REDUCE)) {
            os << " = group->" << v.name << "[";
            os << cg.getVarIndex(getVarAccessDuplication(v.access),
                                 updateSubs[index]);
            os << "]";
        }
        os << ";" << std::endl;
    }

    // Loop through variable references
    for(const auto &v : varRefs) {
        if(v.access == VarAccessMode::READ_ONLY) {
            os << "const ";
        }
       
        os << v.type->getName() << " l" << v.name;

        // If this isn't a reduction, read value from memory
        // **NOTE** by not initialising these variables for reductions, 
        // compilers SHOULD emit a warning if user code doesn't set it to something
        if(!(v.access & VarAccessModeAttribute::REDUCE)) {
            os << " = " << "group->" << v.name << "[";
            os << getVarRefIndex(cg.getArchetype().getVarReferences().at(v.name),
                                 updateSubs[index]);
            os << "]";
        }
        os << ";" << std::endl;
    }
    
    updateSubs.addVarNameSubstitution(cm->getVars(), "", "l");
    updateSubs.addVarNameSubstitution(cm->getVarRefs(), "", "l");
    updateSubs.addParamValueSubstitution(cm->getParamNames(), cg.getArchetype().getParams(),
                                         [&cg](const std::string &p) { return cg.isParamHeterogeneous(p);  },
                                         "", "group->");
    updateSubs.addVarValueSubstitution(cm->getDerivedParams(), cg.getArchetype().getDerivedParams(),
                                       [&cg](const std::string &p) { return cg.isDerivedParamHeterogeneous(p);  },
                                       "", "group->");
    updateSubs.addVarNameSubstitution(cm->getExtraGlobalParams(), "", "group->");

    std::string code = cm->getUpdateCode();
    updateSubs.applyCheckUnreplaced(code, "custom update : merged" + std::to_string(cg.getIndex()));
    //code = ensureFtype(code, modelMerged.getModel().getPrecision());
    os << code;

    // Write read/write variables back to global memory
    for(const auto &v : cm->getVars()) {
        if(v.access & VarAccessMode::READ_WRITE) {
            os << "group->" << v.name << "[";
            os << cg.getVarIndex(getVarAccessDuplication(v.access),
                                 updateSubs[index]);
            os << "] = l" << v.name << ";" << std::endl;
        }
    }

    // Write read/write variable references back to global memory
    for(const auto &v : varRefs) {
        if(v.access == VarAccessMode::READ_WRITE) {
            os << "group->" << v.name << "[";
            os << getVarRefIndex(cg.getArchetype().getVarReferences().at(v.name),
                                 updateSubs[index]);
            os << "] = l" << v.name << ";" << std::endl;
        }
    }
}
}   // Anonymous namespace

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
    GroupMergedTypeEnvironment<CustomUpdateGroupMerged> typeEnvironment(*this);

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
void CustomUpdateGroupMerged::generateCustomUpdate(const BackendBase&, CodeStream &os, Substitutions &popSubs) const
{
    // Build initial environment with ID etc
    // **TODO** this should happen in backend
    EnvironmentSubstitute subs(os);
    subs.addSubstitution("id", popSubs["id"]);
    
    // Create an environment which caches variables in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarAdapter, CustomUpdateInternal> varSubs(
        getArchetype(), subs, 
        [this](const Models::VarInit&, VarAccess a)
        {
            return getVarIndex(getVarAccessDuplication(a), "id");
        });
    
    // Create an environment which caches variable references in local variables if they are accessed
    EnvironmentLocalVarCache<CustomUpdateVarRefAdapter, CustomUpdateInternal> varRefSubs(
        getArchetype(), subs, 
        [this](const Models::VarReference &v, VarAccessMode)
        {
            return getVarRefIndex(v.getDelayNeuronGroup() != nullptr, 
                                    getVarAccessDuplication(v.getVar().access), 
                                    "id");
        });

    // Pretty print previously parsed update statements
    PrettyPrinter::print(m_UpdateStatements, varRefSubs, getTypeContext());
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
    GroupMergedTypeEnvironment<CustomUpdateWUGroupMergedBase> typeEnvironment(*this);

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
}

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateWUGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateWUGroupMerged::name = "CustomUpdateWU";
//----------------------------------------------------------------------------
void CustomUpdateWUGroupMerged::generateCustomUpdate(const BackendBase&, CodeStream &os, Substitutions &popSubs) const
{
    genCustomUpdate(os, popSubs, *this, "id_syn",
                    [this](const auto &varRef, const std::string &index) 
                    {  
                        return getVarRefIndex(getVarAccessDuplication(varRef.getVar().access),
                                              index);
                    });
}

//----------------------------------------------------------------------------
// CustomUpdateTransposeWUGroupMerged
//----------------------------------------------------------------------------
const std::string CustomUpdateTransposeWUGroupMerged::name = "CustomUpdateTransposeWU";
//----------------------------------------------------------------------------
void CustomUpdateTransposeWUGroupMerged::generateCustomUpdate(const BackendBase&, CodeStream &os, Substitutions &popSubs) const
{
    genCustomUpdate(os, popSubs, *this, "id_syn",
                    [this](const auto &varRef, const std::string &index) 
                    {
                        return getVarRefIndex(getVarAccessDuplication(varRef.getVar().access),
                                              index);
                    });
}

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
