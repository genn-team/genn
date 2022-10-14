#pragma once

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityUpdateGroupMerged
//----------------------------------------------------------------------------
namespace CodeGenerator
{
class GENN_EXPORT CustomConnectivityUpdateGroupMerged : public GroupMerged<CustomConnectivityUpdateInternal>
{
public:
    CustomConnectivityUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                        const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(size_t index) const;
    bool isDerivedParamHeterogeneous(size_t index) const;

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

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// CodeGenerator::CustomConnectivityHostUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityHostUpdateGroupMerged : public GroupMerged<CustomConnectivityUpdateInternal>
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
    bool isParamHeterogeneous(size_t index) const;
    bool isDerivedParamHeterogeneous(size_t index) const;

    void addVarPushPullFuncSubs(const BackendBase &backend, Substitutions &subs, 
                                const Models::Base::VarVec &vars, const std::string &count,
                                VarLocation(CustomConnectivityUpdateInternal:: *getVanLocationFn)(size_t) const) const;
};
}   // namespace CodeGenerator