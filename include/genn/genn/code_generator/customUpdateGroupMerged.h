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

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateCustomUpdate(EnvironmentExternalBase &env, unsigned int batchSize,
                              BackendBase::GroupHandlerEnv<CustomUpdateGroupMerged> genPostamble);

    std::string getVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;
    std::string getVarRefIndex(const NeuronGroup *delayNeuronGroup, unsigned int batchSize, VarAccessDim varDims,
                               const std::string &index, const std::string &delaySlot) const;

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

    void generateCustomUpdate(EnvironmentExternalBase &env, unsigned int batchSize,
                              BackendBase::GroupHandlerEnv<CustomUpdateWUGroupMergedBase> genPostamble);

    std::string getVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;
    std::string getVarRefIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;

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
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
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
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    std::string addTransposeField(EnvironmentGroupMergedField<CustomUpdateTransposeWUGroupMerged> &env);

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
    void generateCustomUpdateBase(EnvironmentGroupMergedField<M> &env)
    {
        // Loop through variables and add pointers if they are reduction targets
        const auto *cm = this->getArchetype().getModel();
        for(const auto &v : cm->getVars()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                const auto fieldType = v.type.resolve(this->getTypeContext()).createPointer();
                env.addField(fieldType, v.name, v.name,
                             [v](const auto &runtime, const auto &g, size_t) 
                             {
                                 return runtime.getArray(g, v.name); 
                             });
            }
        }

        // Loop through variable references and add pointers if they are reduction targets
        for(const auto &v : cm->getVarRefs()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                const auto fieldType = v.type.resolve(this->getTypeContext()).createPointer();
                env.addField(fieldType, v.name, v.name,
                             [v](const auto &runtime, const auto &g, size_t) 
                             {
                                 return g.getVarReferences().at(v.name).getTargetArray(runtime);
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
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name, true);
    }

    void generateCustomUpdate(EnvironmentGroupMergedField<CustomUpdateHostReductionGroupMerged> &env);

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
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name, true);
    }

    void generateCustomUpdate(EnvironmentGroupMergedField<CustomWUUpdateHostReductionGroupMerged> &env);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace GeNN::CodeGenerator
