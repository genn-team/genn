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
    //! Child group merged for current sources attached to this neuron update group
    class CurrentSource : public GroupMerged<CurrentSourceInternal>
    {
    public:
        CurrentSource(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                      const std::vector<std::reference_wrapper<const CurrentSourceInternal>> &groups);

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    private:
        //----------------------------------------------------------------------------
        // Private API
        //----------------------------------------------------------------------------
        //! Is the current source parameter referenced?
        bool isParamReferenced(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynPSM
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups
    class InSynPSM : public GroupMerged<SynapseGroupInternal>
    {
    public:
        InSynPSM(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    private:
        //----------------------------------------------------------------------------
        // Private API
        //----------------------------------------------------------------------------
        //! Is the current source parameter referenced?
        bool isParamReferenced(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynPreOutput
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with $(addToPre) logic
    class OutSynPreOutput : public GroupMerged<SynapseGroupInternal>
    {
    public:
        OutSynPreOutput(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                        const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynWUMPostCode
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups with postsynaptic update/spike code
    class InSynWUMPostCode : public GroupMerged<SynapseGroupInternal>
    {
    public:
        InSynWUMPostCode(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                         const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs, bool dynamicsNotSpike) const;

        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    private:
        //----------------------------------------------------------------------------
        // Private API
        //----------------------------------------------------------------------------
        //! Is the current source parameter referenced?
        bool isParamReferenced(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynWUMPreCode
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with presynaptic update/spike code
    class OutSynWUMPreCode : public GroupMerged<SynapseGroupInternal>
    {
    public:
        OutSynWUMPreCode(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                         const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, CodeStream &os, const NeuronUpdateGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs, bool dynamicsNotSpike) const;

        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    private:
        //----------------------------------------------------------------------------
        // Private API
        //----------------------------------------------------------------------------
        //! Is the current source parameter referenced?
        bool isParamReferenced(const std::string &paramName) const;
    };

    NeuronUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
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
    void addNeuronModelSubstitutions(Substitutions &substitution, const std::string &sourceSuffix = "", const std::string &destSuffix = "") const;
    
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<CurrentSource> m_CurrentSources;
    std::vector<InSynPSM> m_InSynPSMs;
    std::vector<OutSynPreOutput> m_OutSynPreOutput;
    std::vector<CurrentSource> m_SortedInSynWithPostCode;
    std::vector<InSynWUMPostCode> m_InSynWUMPostCode;
    std::vector<OutSynWUMPreCode> m_OutSynWUMPreCode;
};
}   // namespace GeNN::CodeGenerator
