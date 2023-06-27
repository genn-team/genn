#pragma once

// Standard C++ includes
#include <tuple>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
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

    void generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged) const;

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

    void generateUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged) const;

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

    void addVarPushPullFuncSubs(const BackendBase &backend, Substitutions &subs, 
                                const Models::Base::VarVec &vars, const std::string &count,
                                VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(const std::string&) const) const;

    void addVars(const BackendBase &backend, const Models::Base::VarVec &vars,
                 VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(const std::string&) const);
};
}   // namespace GeNN::CodeGenerator
