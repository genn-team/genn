#pragma once

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/environment.h"
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateGroupMerged
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT CustomUpdateGroupMerged : public GroupMerged<CustomUpdateInternal>
{
public:
    using GroupMerged::GroupMerged;

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

    void generateCustomUpdate(const BackendBase &backend, EnvironmentExternalBase &env);

    std::string getVarIndex(VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getVarRefIndex(bool delay, VarAccessDuplication varDuplication, const std::string &index) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(const std::string &paramName) const;
    bool isDerivedParamHeterogeneous(const std::string &paramName) const;
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateWUGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWUGroupMergedBase : public GroupMerged<CustomUpdateWUInternal>
{
public:
    using GroupMerged::GroupMerged;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(const std::string &paramName) const;
    bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    std::string getVarIndex(VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getVarRefIndex(VarAccessDuplication varDuplication, const std::string &index) const;

protected:
    void generateCustomUpdateBase(const BackendBase &backend, EnvironmentExternalBase &env,
                                  BackendBase::GroupHandlerEnv<CustomUpdateWUGroupMergedBase> genTranspose);
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateWUGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWUGroupMerged : public CustomUpdateWUGroupMergedBase
{
public:
    using CustomUpdateWUGroupMergedBase::CustomUpdateWUGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateCustomUpdate(const BackendBase &backend, EnvironmentExternalBase &env)
    {
        generateCustomUpdateBase(backend, env, [](auto &env, const auto&){});
    }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateTransposeWUGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateTransposeWUGroupMerged : public CustomUpdateWUGroupMergedBase
{
public:
    using CustomUpdateWUGroupMergedBase::CustomUpdateWUGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateCustomUpdate(const BackendBase &backend, EnvironmentExternalBase &env,
                              BackendBase::GroupHandlerEnv<CustomUpdateWUGroupMergedBase> genTranspose)
    {
       generateCustomUpdateBase(backend, env, genTranspose);
    }

    std::string addTransposeField(const BackendBase &backend, EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> &env);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateHostReductionGroupMergedBase
//----------------------------------------------------------------------------
template<typename G>
class CustomUpdateHostReductionGroupMergedBase : public GroupMerged<G>
{
protected:
    using GroupMerged<G>::GroupMerged;

    template<typename M>
    void generateCustomUpdateBase(const BackendBase &backend, EnvironmentGroupMergedField<M> &env)
    {
        // Loop through variables and add pointers if they are reduction targets
        const auto *cm = this->getArchetype().getCustomUpdateModel();
        for(const auto &v : cm->getVars()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                const auto fieldType = v.type.resolve(this->getTypeContext()).createPointer();
                env.addField(fieldType, v.name, v.name,
                             [&backend, v](const auto &g, size_t) 
                             {
                                 return backend.getDeviceVarPrefix() + v.name + g.getName(); 
                             });
            }
        }

        // Loop through variable references and add pointers if they are reduction targets
        for(const auto &v : cm->getVarRefs()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                const auto fieldType = v.type.resolve(this->getTypeContext()).createPointer();
                env.addField(fieldType, v.name, v.name,
                             [&backend, v](const auto &g, size_t) 
                             {
                                 const auto varRef = g.getVarReferences().at(v.name);
                                 return backend.getDeviceVarPrefix() + v.name + varRef.getTargetName(); 
                             });
            }
        }
    }
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateHostReductionGroupMerged : public CustomUpdateHostReductionGroupMergedBase<CustomUpdateInternal>
{
public:
    using CustomUpdateHostReductionGroupMergedBase::CustomUpdateHostReductionGroupMergedBase;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

    void generateCustomUpdate(const BackendBase &backend, EnvironmentGroupMergedField<CustomUpdateHostReductionGroupMerged> &env);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomWUUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateHostReductionGroupMerged : public CustomUpdateHostReductionGroupMergedBase<CustomUpdateWUInternal>
{
public:
    using CustomUpdateHostReductionGroupMergedBase::CustomUpdateHostReductionGroupMergedBase;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

    void generateCustomUpdate(const BackendBase &backend, EnvironmentGroupMergedField<CustomWUUpdateHostReductionGroupMerged> &env);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace GeNN::CodeGenerator
