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
    class GENN_EXPORT CurrentSource : public ChildGroupMerged<CurrentSourceInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, 
                      unsigned int batchSize);

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynPSM
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups
    class GENN_EXPORT InSynPSM : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env,
                      NeuronUpdateGroupMerged &ng, unsigned int batchSize);

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynPreOutput
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with $(addToPre) logic
    class GENN_EXPORT OutSynPreOutput : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                      unsigned int batchSize);
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::SynSpike
    //----------------------------------------------------------------------------
    //! Child group merged for synapse groups that process spikes or spike-events
    /*! There is no generic code to generate here as this is backend-specific */
    class GENN_EXPORT SynSpike : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                      BackendBase::HandlerEnv genUpdate);
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::SynSpikeEvent
    //----------------------------------------------------------------------------
    //! Child group merged for synapse groups that process spikes or spike-events
    /*! There is no generic code to generate here as this is backend-specific */
    class GENN_EXPORT SynSpikeEvent : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                      BackendBase::GroupHandlerEnv<SynSpikeEvent> genUpdate);

        void generateEventCondition(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                    unsigned int batchSize, BackendBase::GroupHandlerEnv<SynSpikeEvent> genEmitSpikeLikeEvent);

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;
    
        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;
    private:
        void generateEventConditionInternal(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                            unsigned int batchSize, BackendBase::GroupHandlerEnv<SynSpikeEvent> genEmitSpikeLikeEvent,
                                            const std::vector<Transpiler::Token> &conditionTokens, const std::string &errorContext);

    };


    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynWUMPostCode
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups with postsynaptic update/spike code
    class GENN_EXPORT InSynWUMPostCode : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                      unsigned int batchSize, bool dynamicsNotSpike);

        void genCopyDelayedVars(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng, 
                                unsigned int batchSize);

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynWUMPreCode
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with presynaptic update/spike code
    class GENN_EXPORT OutSynWUMPreCode : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                      unsigned int batchSize, bool dynamicsNotSpike);

        void genCopyDelayedVars(EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                                unsigned int batchSize);

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the current source parameter be implemented heterogeneously?
        bool isParamHeterogeneous(const std::string &paramName) const;

        //! Should the current source derived parameter be implemented heterogeneously?
        bool isDerivedParamHeterogeneous(const std::string &paramName) const;
    };

    NeuronUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
                            const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get hash digest used for detecting changes
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }
    
    void generateNeuronUpdate(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize,
                              BackendBase::HandlerEnv genEmitTrueSpike,
                              BackendBase::GroupHandlerEnv<SynSpikeEvent> genEmitSpikeLikeEvent);
    
    void generateSpikes(EnvironmentExternalBase &env, BackendBase::HandlerEnv genUpdate);
    void generateSpikeEvents(EnvironmentExternalBase &env, BackendBase::GroupHandlerEnv<SynSpikeEvent> genUpdate);
    
    void generateWUVarUpdate(EnvironmentExternalBase &env, unsigned int batchSize);
    
    std::string getVarIndex(unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;
    std::string getReadVarIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;
    std::string getWriteVarIndex(bool delay, unsigned int batchSize, VarAccessDim varDims, const std::string &index) const;

    const std::vector<CurrentSource> &getMergedCurrentSourceGroups() const { return m_MergedCurrentSourceGroups; }
    const std::vector<InSynPSM> &getMergedInSynPSMGroups() const { return m_MergedInSynPSMGroups; }
    const std::vector<OutSynPreOutput> &getMergedOutSynPreOutputGroups() const { return m_MergedOutSynPreOutputGroups; }
    const std::vector<SynSpike> &getMergedSpikeGroups() const{ return m_MergedSpikeGroups; }
    const std::vector<SynSpikeEvent> &getMergedSpikeEventGroups() const{ return m_MergedSpikeEventGroups; }
    const std::vector<InSynWUMPostCode> &getMergedInSynWUMPostCodeGroups() const { return m_MergedInSynWUMPostCodeGroups; }
    const std::vector<OutSynWUMPreCode> &getMergedOutSynWUMPreCodeGroups() const { return m_MergedOutSynWUMPreCodeGroups; }
    
    //! Should the parameter be implemented heterogeneously?
    bool isParamHeterogeneous(const std::string &paramName) const;

    //! Should the derived parameter be implemented heterogeneously?
    bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<CurrentSource> m_MergedCurrentSourceGroups;
    std::vector<InSynPSM> m_MergedInSynPSMGroups;
    std::vector<OutSynPreOutput> m_MergedOutSynPreOutputGroups;
    std::vector<SynSpike> m_MergedSpikeGroups;
    std::vector<SynSpikeEvent> m_MergedSpikeEventGroups;
    std::vector<InSynWUMPostCode> m_MergedInSynWUMPostCodeGroups;
    std::vector<OutSynWUMPreCode> m_MergedOutSynWUMPreCodeGroups;
};


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronSpikeQueueUpdateGroupMerged : public NeuronGroupMergedBase
{
public:
    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::SynSpike
    //----------------------------------------------------------------------------
    //! Child group merged for synapse groups that process spikes
    /*! There is no generic code to generate here as this is backend-specific */
    class SynSpike : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronSpikeQueueUpdateGroupMerged &ng,
                      unsigned int batchSize);
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged::SynSpikeEvent
    //----------------------------------------------------------------------------
    //! Child group merged for synapse groups that process spikes events
    /*! There is no generic code to generate here as this is backend-specific */
    class SynSpikeEvent : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronSpikeQueueUpdateGroupMerged &ng,
                      unsigned int batchSize);
    };

    NeuronSpikeQueueUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                      const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    const std::vector<SynSpike> &getMergedSpikeGroups() const{ return m_MergedSpikeGroups; }
    const std::vector<SynSpikeEvent> &getMergedSpikeEventGroups() const{ return m_MergedSpikeEventGroups; }
    
    void genSpikeQueueUpdate(EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<SynSpike> m_MergedSpikeGroups;
    std::vector<SynSpikeEvent> m_MergedSpikeEventGroups;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronPrevSpikeTimeUpdateGroupMerged : public NeuronGroupMergedBase
{
public:
    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged::SynSpike
    //----------------------------------------------------------------------------
    //! Child group merged for synapse groups that require previous spike times
    class SynSpike : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng,
                      BackendBase::HandlerEnv genUpdate);
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged::SynSpikeEvent
    //----------------------------------------------------------------------------
    //! Child group merged for synapse groups that require previous spike event times
    class SynSpikeEvent : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(EnvironmentExternalBase &env, NeuronPrevSpikeTimeUpdateGroupMerged &ng,
                      BackendBase::HandlerEnv genUpdate);
    };

    NeuronPrevSpikeTimeUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                         const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateSpikes(EnvironmentExternalBase &env, BackendBase::HandlerEnv genUpdate);
    void generateSpikeEvents(EnvironmentExternalBase &env, BackendBase::HandlerEnv genUpdate);
    

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<SynSpike> m_MergedSpikeGroups;
    std::vector<SynSpikeEvent> m_MergedSpikeEventGroups;
};
}   // namespace GeNN::CodeGenerator
