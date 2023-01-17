#pragma once

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateGroupMerged
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT CustomUpdateGroupMerged : public GroupMerged<CustomUpdateInternal>
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

    void generateRunner(const BackendBase &backend, const Type::TypeContext &context, 
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, context, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    std::string getVarIndex(VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getVarRefIndex(bool delay, VarAccessDuplication varDuplication, const std::string &index) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateWUGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWUGroupMergedBase : public GroupMerged<CustomUpdateWUInternal>
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(const std::string &paramName) const;
    bool isDerivedParamHeterogeneous(const std::string &paramName) const;

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
// GeNN::CodeGenerator::CustomUpdateWUGroupMerged
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
    void generateRunner(const BackendBase &backend, const Type::TypeContext &context, 
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, context, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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
    CustomUpdateTransposeWUGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
        : CustomUpdateWUGroupMergedBase(index, precision, timePrecision, backend, groups)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, const Type::TypeContext &context, 
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, context, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateCustomUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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
     CustomUpdateHostReductionGroupMergedBase(size_t index, const std::string &precision, const BackendBase &backend,
                                   const std::vector<std::reference_wrapper<const G>> &groups)
    :   GroupMerged<G>(index, precision, groups)
    {
        // Create type environment
        // **TEMP** parse precision to get scalar type
        //GroupMergedTypeEnvironment<CustomUpdateHostReductionGroupMergedBase> typeEnvironment(*this, Type::parseNumeric(precision));
        
        // Loop through variables and add pointers if they are reduction targets
        const CustomUpdateModels::Base *cm = this->getArchetype().getCustomUpdateModel();
        for(const auto &v : cm->getVars()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                this->addPointerField(Type::parseNumeric(v.type), v.name, backend.getDeviceVarPrefix() + v.name);
            }
        }

        // Loop through variable references and add pointers if they are reduction targets
        for(const auto &v : cm->getVarRefs()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                this->addPointerField(Type::parseNumeric(v.type), v.name, backend.getDeviceVarPrefix() + v.name);
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
    CustomUpdateHostReductionGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                         const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, const Type::TypeContext &context, 
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, context, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

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
    CustomWUUpdateHostReductionGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                           const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, const Type::TypeContext &context, 
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, context, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace GeNN::CodeGenerator
