#pragma once

// Standard C++ includes
#include <tuple>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityUpdateGroupMergedBase
//----------------------------------------------------------------------------
namespace CodeGenerator
{
class GENN_EXPORT CustomConnectivityUpdateGroupMergedBase : public GroupMerged<CustomConnectivityUpdateInternal>
{
public:
    CustomConnectivityUpdateGroupMergedBase(size_t index, const std::string &precision, const BackendBase &backend,
                                            const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups);


    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(size_t index) const;
    bool isDerivedParamHeterogeneous(size_t index) const;

protected:

};

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityUpdateGroupMerged : public CustomConnectivityUpdateGroupMergedBase
{
public:
    CustomConnectivityUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                        const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                            runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    std::string getPrePostVarRefIndex(bool delay, const std::string &index) const;

    //! Get sorted vector of variable names, types and duplication modes which 
    //! need updating when synapses are added and removed, belonging to archetype group
    const std::vector<Models::WUVarReference> &getSortedArchetypeDependentVars() const { return m_SortedDependentVars.front(); }
    
    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    //! Sorted vectors of variable names, types and duplication modes which 
    //! need updating when synapses are added and removed to each group
    std::vector<std::vector<Models::WUVarReference>> m_SortedDependentVars;
};

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityHostUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityHostUpdateGroupMerged : public CustomConnectivityUpdateGroupMergedBase
{
public:
    CustomConnectivityHostUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                            const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups);
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                            runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

    void generateUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    void addVarPushPullFuncSubs(const BackendBase &backend, Substitutions &subs, 
                                const Models::Base::VarVec &vars, const std::string &count,
                                VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(size_t) const) const;

    void addVars(const BackendBase &backend, const Models::Base::VarVec &vars,
                 VarLocation(CustomConnectivityUpdateInternal:: *getVarLocationFn)(size_t) const);
};
}   // namespace CodeGenerator
