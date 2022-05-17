#pragma once

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateGroupMerged
//----------------------------------------------------------------------------
namespace CodeGenerator
{
class GENN_EXPORT CustomUpdateGroupMerged : public GroupMerged<CustomUpdateInternal>
{
public:
    CustomUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

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

    void generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    std::string getVarIndex(VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getVarRefIndex(bool delay, VarAccessDuplication varDuplication, const std::string &index) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateWUGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWUGroupMergedBase : public GroupMerged<CustomUpdateWUInternal>
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(size_t index) const;
    bool isDerivedParamHeterogeneous(size_t index) const;

    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    std::string getVarIndex(VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getVarRefIndex(VarAccessDuplication varDuplication, const std::string &index) const;

    //! Is kernel size heterogeneous in this dimension?
    bool isKernelSizeHeterogeneous(size_t dimensionIndex) const
    {
        return CodeGenerator::isKernelSizeHeterogeneous(this, dimensionIndex, getGroupKernelSize);
    }

    //! Get expression for kernel size in dimension (may be literal or group->kernelSizeXXX)
    std::string getKernelSize(size_t dimensionIndex) const
    {
        return CodeGenerator::getKernelSize(this, dimensionIndex, getGroupKernelSize);
    }

    //! Generate an index into a kernel based on the id_kernel_XXX variables in subs
    void genKernelIndex(std::ostream& os, const CodeGenerator::Substitutions& subs) const
    {
        return CodeGenerator::genKernelIndex(this, os, subs, getGroupKernelSize);
    }

protected:
    CustomUpdateWUGroupMergedBase(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                  const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

private:
    static const std::vector<unsigned int>& getGroupKernelSize(const CustomUpdateWUInternal& g)
    {
        return g.getSynapseGroup()->getKernelSize();
    }
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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace CodeGenerator