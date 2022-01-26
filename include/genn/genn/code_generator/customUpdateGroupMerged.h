#pragma once

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateGroupMerged
//----------------------------------------------------------------------------
namespace CodeGenerator
{
class GENN_EXPORT CustomUpdateGroupMerged : public RuntimeGroupMerged<CustomUpdateInternal>
{
public:
    CustomUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(const std::string &paramName) const;
    bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    std::string getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getVarRefIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateWUGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWUGroupMergedBase : public RuntimeGroupMerged<CustomUpdateWUInternal>
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(const std::string &paramName) const;
    bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    std::string getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getVarRefIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;

protected:
    CustomUpdateWUGroupMergedBase(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                  const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    void addSynapseGroupPointerField(const std::string &type, const std::string &name);
};

// ----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateWUGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWUGroupMerged : public CustomUpdateWUGroupMergedBase
{
public:
    CustomUpdateWUGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                              const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
        : CustomUpdateWUGroupMergedBase(index, precision, timePrecision, backend, groups)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateTransposeWUGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateTransposeWUGroupMerged : public CustomUpdateWUGroupMergedBase
{
public:
    CustomUpdateTransposeWUGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
        : CustomUpdateWUGroupMergedBase(index, precision, timePrecision, backend, groups)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace CodeGenerator