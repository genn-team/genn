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
    class CurrentSource : public ChildGroupMerged<CurrentSourceInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronUpdateGroupMerged &ng, const ModelSpecMerged &modelMerged);

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
        //! Is the parameter referenced? **YUCK** only used for hashing
        bool isParamReferenced(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynPSM
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups
    class InSynPSM : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env,
                      NeuronUpdateGroupMerged &ng, const ModelSpecMerged &modelMerged);

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
        //! Is the parameter referenced? **YUCK** only used for hashing
        bool isParamReferenced(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynPreOutput
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with $(addToPre) logic
    class OutSynPreOutput : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronUpdateGroupMerged &ng, const ModelSpecMerged &modelMerged);
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::InSynWUMPostCode
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups with postsynaptic update/spike code
    class InSynWUMPostCode : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, bool dynamicsNotSpike);

        void genCopyDelayedVars(EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng,
                                const ModelSpecMerged &modelMerged);

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
        //! Is the parameter referenced? **YUCK** only used for hashing
        bool isParamReferenced(const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronUpdateGroupMerged::OutSynWUMPreCode
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with presynaptic update/spike code
    class OutSynWUMPreCode : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, NeuronUpdateGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, bool dynamicsNotSpike);

        void genCopyDelayedVars(EnvironmentExternalBase &env, const NeuronUpdateGroupMerged &ng,
                                const ModelSpecMerged &modelMerged);

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
        //! Is the parameter referenced? **YUCK** only used for hashing
        bool isParamReferenced(const std::string &paramName) const;
    };

    NeuronUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext,
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
    
    void generateNeuronUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged,
                              BackendBase::GroupHandlerEnv<NeuronUpdateGroupMerged> genEmitTrueSpike,
                              BackendBase::GroupHandlerEnv<NeuronUpdateGroupMerged> genEmitSpikeLikeEvent);
    
    void generateWUVarUpdate(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);
    
    std::string getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getReadVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getWriteVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;

    const std::vector<CurrentSource> &getMergedCurrentSourceGroups() const { return m_MergedCurrentSourceGroups; }
    const std::vector<InSynPSM> &getMergedInSynPSMGroups() const { return m_MergedInSynPSMGroups; }
    const std::vector<OutSynPreOutput> &getMergedOutSynPreOutputGroups() const { return m_MergedOutSynPreOutputGroups; }
    const std::vector<InSynWUMPostCode> &getMergedInSynWUMPostCodeGroups() const { return m_MergedInSynWUMPostCodeGroups; }
    const std::vector<OutSynWUMPreCode> &getMergedOutSynWUMPreCodeGroups() const { return m_MergedOutSynWUMPreCodeGroups; }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
     //! Should the parameter be implemented heterogeneously?
    bool isParamHeterogeneous(const std::string &paramName) const;

    //! Should the derived parameter be implemented heterogeneously?
    bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<CurrentSource> m_MergedCurrentSourceGroups;
    std::vector<InSynPSM> m_MergedInSynPSMGroups;
    std::vector<OutSynPreOutput> m_MergedOutSynPreOutputGroups;
    std::vector<InSynWUMPostCode> m_MergedInSynWUMPostCodeGroups;
    std::vector<OutSynWUMPreCode> m_MergedOutSynWUMPreCodeGroups;
};
}   // namespace GeNN::CodeGenerator
