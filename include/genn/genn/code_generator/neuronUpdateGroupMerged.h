#pragma once

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT NeuronUpdateGroupMerged : public NeuronGroupMergedBase
{
public:
    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::CurrentSource
    //----------------------------------------------------------------------------
    class CurrentSource : public GroupMerged<CurrentSourceInternal>
    {
    public:
        CurrentSource(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                      const std::vector<std::reference_wrapper<const CurrentSourceInternal>> &groups);

        // Public API
        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    private:
        //! Is the current source parameter referenced?
        bool isParamReferenced(const std::string &paramName) const;
    };

    NeuronUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Should the incoming synapse weight update model parameter be implemented heterogeneously?
    bool isInSynWUMParamHeterogeneous(size_t childIndex, const std::string &paramName) const;

    //! Should the incoming synapse weight update model derived parameter be implemented heterogeneously?
    bool isInSynWUMDerivedParamHeterogeneous(size_t childIndex, const std::string &paramName) const;

    //! Should the outgoing synapse weight update model parameter be implemented heterogeneously?
    bool isOutSynWUMParamHeterogeneous(size_t childIndex, const std::string &paramName) const;

    //! Should the outgoing synapse weight update model derived parameter be implemented heterogeneously?
    bool isOutSynWUMDerivedParamHeterogeneous(size_t childIndex, const std::string &paramName) const;

    //! Get sorted vectors of incoming synapse groups with postsynaptic code belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeInSynWithPostCode() const { return m_SortedInSynWithPostCode.front(); }

    //! Get sorted vectors of outgoing synapse groups with presynaptic code belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeOutSynWithPreCode() const { return m_SortedOutSynWithPreCode.front(); }

    //! Get hash digest used for detecting changes
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, 
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }
    
    void generateNeuronUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs,
                              BackendBase::GroupHandler<NeuronUpdateGroupMerged> genEmitTrueSpike,
                              BackendBase::GroupHandler<NeuronUpdateGroupMerged> genEmitSpikeLikeEvent) const;
    
    void generateWUVarUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    
    std::string getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getReadVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getWriteVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    //! Helper to generate merged struct fields for WU pre and post vars
    void generateWUVar(const BackendBase &backend, const std::string &fieldPrefixStem, 
                       const std::vector<std::vector<SynapseGroupInternal*>> &sortedSyn,
                       Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                       bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, const std::string&) const,
                       bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, const std::string&) const,
                       const std::string&(SynapseGroupInternal::*getFusedVarSuffix)(void) const);

    //! Is the incoming synapse weight update model parameter referenced?
    bool isInSynWUMParamReferenced(size_t childIndex, const std::string &paramName) const;

    //! Is the outgoing synapse weight update model parameter referenced?
    bool isOutSynWUMParamReferenced(size_t childIndex, const std::string &paramName) const;

    void addNeuronModelSubstitutions(Substitutions &substitution, const std::string &sourceSuffix = "", const std::string &destSuffix = "") const;
    
    void generateWUVarUpdate(CodeStream &os, const Substitutions &popSubs,
                             const std::string &fieldPrefixStem, const std::string &sourceSuffix, 
                             bool useLocalNeuronVars, unsigned int batchSize, 
                             const std::vector<SynapseGroupInternal*> &archetypeSyn,
                             unsigned int(SynapseGroupInternal::*getDelaySteps)(void) const,
                             Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                             std::string(WeightUpdateModels::Base::*getCode)(void) const,
                             bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, const std::string&) const,
                             bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, const std::string&) const) const;
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedInSynWithPostCode;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedOutSynWithPreCode;
};
}   // namespace GeNN::CodeGenerator
