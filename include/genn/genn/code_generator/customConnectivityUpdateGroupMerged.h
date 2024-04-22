#pragma once

// Standard C++ includes
#include <tuple>

// GeNN includes
#include "customConnectivityUpdateInternal.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/environment.h"
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomConnectivityUpdateGroupMerged
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT CustomConnectivityUpdateGroupMerged : public GroupMerged<CustomConnectivityUpdateInternal>
{
public:
    CustomConnectivityUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                        const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //! Get sorted vector of variable names, types and duplication modes which 
    //! need updating when synapses are added and removed, belonging to archetype group
    const std::vector<Models::WUVarReference> &getSortedArchetypeDependentVars() const { return m_SortedDependentVars.front(); }
    
    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(const std::string &name) const;
    bool isDerivedParamHeterogeneous(const std::string &name) const;

    template<typename A>
    void addPrivateVarRefAccess(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env, unsigned int batchSize, 
                                std::function<std::string(VarAccessMode, const typename A::RefType&)> getIndexFn)
    {
        // Loop through variable references
        const A archetypeAdaptor(getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // If model isn't batched or variable isn't duplicated
            const auto &varRef = archetypeAdaptor.getInitialisers().at(v.name);
            if(batchSize == 1 || !(varRef.getVarDims() & VarAccessDim::BATCH)) {
                // Add field with qualified type which indexes private pointer field
                const auto resolvedType = v.type.resolve(getTypeContext());
                const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
                env.add(qualifiedType, v.name, "$(_" + v.name + ")[" + getIndexFn(v.access, varRef) + "]");
            }
        }
    }
    
    template<typename A>
    void addPrivateVarRefAccess(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env, unsigned int batchSize, const std::string &indexSuffix)
    {
        addPrivateVarRefAccess<A>(env, batchSize, [&indexSuffix](VarAccessMode, const typename A::RefType&){ return indexSuffix; });
    }
    
    template<typename A>
    void addPrivateVarAccess(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env, unsigned int batchSize, const std::string &indexSuffix)
    {
        // Loop through variable references
        const A archetypeAdaptor(getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // Add field with qualified type which indexes private pointer field
            const auto resolvedType = v.type.resolve(getTypeContext());
            const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
            env.add(qualifiedType, v.name, "$(_" + v.name + ")[" + indexSuffix + "]");
        }
    }
    
    template<typename V>
    void addTypes(GeNN::Transpiler::TypeChecker::EnvironmentBase &env, const std::vector<V> &vars, 
                  GeNN::Transpiler::ErrorHandlerBase &errorHandler, bool readOnly = false)
    {
        // Loop through variables
        for(const auto &v : vars) {
            const auto resolvedType = v.type.resolve(getTypeContext());
            const auto qualifiedType = (readOnly || (v.access & VarAccessModeAttribute::READ_ONLY)) ? resolvedType.addConst() : resolvedType;
            env.define(Transpiler::Token{Transpiler::Token::Type::IDENTIFIER, v.name, 0}, qualifiedType, errorHandler);
        }
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    //! Sorted vectors of variable names, types and duplication modes which 
    //! need updating when synapses are added and removed to each group
    std::vector<std::vector<Models::WUVarReference>> m_SortedDependentVars;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomConnectivityRemapUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityRemapUpdateGroupMerged : public GroupMerged<CustomConnectivityUpdateInternal>
{
public:
    using GroupMerged::GroupMerged;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase& backend, CodeStream& definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomConnectivityHostUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityHostUpdateGroupMerged : public GroupMerged<CustomConnectivityUpdateInternal>
{
public:
    using GroupMerged::GroupMerged;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name, true);
    }

    void generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(const std::string &name) const;
    bool isDerivedParamHeterogeneous(const std::string &name) const;

    template<typename A>
    void addVars(EnvironmentGroupMergedField<CustomConnectivityHostUpdateGroupMerged> &env, const std::string &count, const BackendBase &backend)
    {
        // Loop through variables and add private pointer field 
        const A archetypeAdaptor(getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // If var is located on the host
            const auto loc = archetypeAdaptor.getLoc(v.name);
            if (loc & VarLocationAttribute::HOST) {
                // Add pointer field to allow user code to access
                const auto resolvedType = v.type.resolve(getTypeContext());
                const auto pointerType = resolvedType.createPointer();
                env.addField(pointerType, "_" + v.name, v.name,
                             [v](const auto &runtime, const auto &g, size_t) 
                             { 
                                 return runtime.getArray(g, v.name);
                             },
                             "", GroupMergedFieldType::HOST);

                // Add substitution for direct access to field
                env.add(pointerType, v.name, "$(_" + v.name + ")");

                 // If backend requires seperate device objects, add additional (private) field)
                if(backend.isArrayDeviceObjectRequired()) {
                    env.addField(pointerType, "_d_" + v.name, "d_" + v.name,
                                 [v](const auto &runtime, const auto &g, size_t)
                                 {
                                     return runtime.getArray(g, v.name);
                                 });
                }

                // Generate code to push this variable
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genLazyVariableDynamicPush(push, resolvedType, v.name,
                                                   loc, count);

                // Add substitution
                env.add(Type::PushPull, "push" + v.name + "ToDevice", pushStream.str());

                // Generate code to pull this variable
                std::stringstream pullStream;
                CodeStream pull(pullStream);
                backend.genLazyVariableDynamicPull(pull, resolvedType, v.name,
                                                   loc, count);

                // Add substitution
                env.add(Type::PushPull, "pull" + v.name + "FromDevice", pullStream.str());
            }
        }
    }
};
}   // namespace GeNN::CodeGenerator
