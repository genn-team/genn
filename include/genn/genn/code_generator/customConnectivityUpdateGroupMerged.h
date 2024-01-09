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

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
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
    void addPrivateVarPointerFields(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env, const std::string &arrayPrefix)
    {
        // Loop through variables and add private pointer field 
        const A archetypeAdaptor(getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(getTypeContext());
            env.addField(resolvedType.createPointer(), "_" + v.name, v.name,
                         [arrayPrefix, v](const auto &g, size_t) 
                         { 
                             return arrayPrefix + v.name + A(g).getNameSuffix();
                         });
        }
    }

    template<typename A>
    void addPrivateVarRefPointerFields(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env, const std::string &arrayPrefix)
    {
        // Loop through variable references and add private pointer field 
        const A archetypeAdaptor(getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            const auto resolvedType = v.type.resolve(getTypeContext());
            env.addField(resolvedType.createPointer(), "_" + v.name, v.name,
                         [arrayPrefix, v](const auto &g, size_t) 
                         { 
                             const auto varRef = A(g).getInitialisers().at(v.name);
                             return arrayPrefix + varRef.getVar().name + varRef.getTargetName(); 
                         });
        }
    }
    
    template<typename A>
    void addPrivateVarRefAccess(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env, unsigned int batchSize, 
                                std::function<std::string(VarAccessMode, const typename A::RefType&)> getIndexFn)
    {
        // Loop through variable references
        const A archetypeAdaptor(getArchetype());
        for(const auto &v : archetypeAdaptor.getDefs()) {
            // If model isn't batched or variable isn't duplicated
            const auto &varRef = archetypeAdaptor.getInitialisers().at(v.name);
            if(batchSize == 1 || !varRef.isDuplicated()) {
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
                  GeNN::Transpiler::ErrorHandlerBase &errorHandler)
    {
        // Loop through variables
        for(const auto &v : vars) {
            const auto resolvedType = v.type.resolve(getTypeContext());
            const auto qualifiedType = (v.access & VarAccessModeAttribute::READ_ONLY) ? resolvedType.addConst() : resolvedType;
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
// GeNN::CodeGenerator::CustomConnectivityHostUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityHostUpdateGroupMerged : public GroupMerged<CustomConnectivityUpdateInternal>
{
public:
    using GroupMerged::GroupMerged;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
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
            if (loc & VarLocation::HOST) {
                // Add pointer field to allow user code to access
                const auto resolvedType = v.type.resolve(getTypeContext());
                const auto pointerType = resolvedType.createPointer();
                env.addField(pointerType, "_" + v.name, v.name,
                             [v](const auto &g, size_t) 
                             { 
                                 return  v.name + g.getName();
                             },
                             "", GroupMergedFieldType::HOST);

                // Add substitution for direct access to field
                env.add(pointerType, v.name, "$(_" + v.name + ")");

                // If backend has device variables, also add hidden pointer field with device pointer
                if(!backend.getDeviceVarPrefix().empty())  {
                    env.addField(pointerType, "_" + backend.getDeviceVarPrefix() + v.name, backend.getDeviceVarPrefix() + v.name,
                                 [v, &backend](const auto &g, size_t)
                                 {
                                     return backend.getDeviceVarPrefix() + v.name + g.getName();
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
