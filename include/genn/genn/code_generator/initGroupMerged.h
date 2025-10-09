#pragma once

// GeNN includes
#include "customConnectivityUpdateInternal.h"
#include "currentSourceInternal.h"
#include "customUpdateInternal.h"
#include "synapseGroupInternal.h"

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::InitGroupMergedBase
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
template<typename B, typename A>
class InitGroupMergedBase : public B
{
public:
    using B::B;

protected:
    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    void updateBaseHash(boost::uuids::detail::sha1 &hash) const
    {
        // Update hash with each group's variable initialisation parameters and derived parameters
        this->template updateVarInitParamHash<A>(hash);

        this->template updateVarInitDerivedParamHash<A>(hash);
    }
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronInitGroupMerged : public InitGroupMergedBase<NeuronGroupMergedBase, NeuronVarAdapter>
{
public:
    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::CurrentSource
    //----------------------------------------------------------------------------
    //! Child group merged for current sources attached to this neuron update group
    class CurrentSource : public InitGroupMergedBase<ChildGroupMerged<CurrentSourceInternal>, CurrentSourceVarAdapter>
    {
    public:
        using InitGroupMergedBase::InitGroupMergedBase;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronInitGroupMerged &ng, unsigned int batchSize);
    
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const
        {
            updateBaseHash(hash);
            Utils::updateHash(getArchetype().getInitHashDigest(getArchetype().getTrgNeuronGroup()), hash);

        }
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::SynSpike
    //----------------------------------------------------------------------------
    //! Child group merged for synapse groups that process spikes
    class SynSpike : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronInitGroupMerged &ng, unsigned int batchSize);
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::SynSpikeEvent
    //----------------------------------------------------------------------------
    //! Child group merged for synapse groups that process spikes events
    class SynSpikeEvent : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronInitGroupMerged &ng, unsigned int batchSize);
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::InSynPSM
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups
    class InSynPSM : public InitGroupMergedBase<ChildGroupMerged<SynapseGroupInternal>, SynapsePSMVarAdapter>
    {
    public:
       using InitGroupMergedBase::InitGroupMergedBase;

       //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronInitGroupMerged &ng, unsigned int batchSize);
        
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const
        {
            updateBaseHash(hash);
            Utils::updateHash(getArchetype().getPSInitHashDigest(getArchetype().getTrgNeuronGroup()), hash);
        }
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynPreOutput
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
                      NeuronInitGroupMerged &ng, unsigned int batchSize);
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::InSynWUMPostCode
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups with postsynaptic variables
    class InSynWUMPostVars : public InitGroupMergedBase<ChildGroupMerged<SynapseGroupInternal>, SynapseWUPostVarAdapter>
    {
    public:
        using InitGroupMergedBase::InitGroupMergedBase;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronInitGroupMerged &ng, unsigned int batchSize);

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const
        {
            updateBaseHash(hash);
            Utils::updateHash(getArchetype().getWUPrePostInitHashDigest(getArchetype().getTrgNeuronGroup()), hash);
        }
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynWUMPreVars
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with presynaptic variables
    class OutSynWUMPreVars: public InitGroupMergedBase<ChildGroupMerged<SynapseGroupInternal>, SynapseWUPreVarAdapter>
    {
    public:
        using InitGroupMergedBase::InitGroupMergedBase;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronInitGroupMerged &ng, unsigned int batchSize);
        
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const
        {
            updateBaseHash(hash);
            Utils::updateHash(getArchetype().getWUPrePostInitHashDigest(getArchetype().getSrcNeuronGroup()), hash);
        }
    };

    NeuronInitGroupMerged(size_t index, const Type::TypeContext &typeContext,
                          const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Get hash digest used for detecting changes
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    const std::vector<CurrentSource> &getMergedCurrentSourceGroups() const { return m_MergedCurrentSourceGroups; }
    const std::vector<SynSpike> &getMergedSpikeGroups() const{ return m_MergedSpikeGroups; }
    const std::vector<SynSpikeEvent> &getMergedSpikeEventGroups() const{ return m_MergedSpikeEventGroups; }
    const std::vector<InSynPSM> &getMergedInSynPSMGroups() const { return m_MergedInSynPSMGroups; }
    const std::vector<OutSynPreOutput> &getMergedOutSynPreOutputGroups() const { return m_MergedOutSynPreOutputGroups; }
    const std::vector<InSynWUMPostVars> &getMergedInSynWUMPostVarGroups() const { return m_MergedInSynWUMPostVarGroups; }
    const std::vector<OutSynWUMPreVars> &getMergedOutSynWUMPreVarGroups() const { return m_MergedOutSynWUMPreVarGroups; }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<CurrentSource> m_MergedCurrentSourceGroups;
    std::vector<SynSpike> m_MergedSpikeGroups;
    std::vector<SynSpikeEvent> m_MergedSpikeEventGroups;
    std::vector<InSynPSM> m_MergedInSynPSMGroups;
    std::vector<OutSynPreOutput> m_MergedOutSynPreOutputGroups;
    std::vector<InSynWUMPostVars> m_MergedInSynWUMPostVarGroups;
    std::vector<OutSynWUMPreVars> m_MergedOutSynWUMPreVarGroups;
};


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseInitGroupMerged : public InitGroupMergedBase<GroupMerged<SynapseGroupInternal>, SynapseWUVarAdapter>
{
public:
    using InitGroupMergedBase::InitGroupMergedBase;

    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseSparseInitGroupMerged : public InitGroupMergedBase<GroupMerged<SynapseGroupInternal>, SynapseWUVarAdapter>
{
public:
    using InitGroupMergedBase::InitGroupMergedBase;

    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityInitGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    using GroupMerged::GroupMerged;
    
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateSparseRowInit(const BackendBase &backend, EnvironmentExternalBase &env);
    void generateSparseColumnInit(const BackendBase &backend, EnvironmentExternalBase &env);
    void generateKernelInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    //! Generate either row or column connectivity init code
    void genInitConnectivity(const BackendBase &backend, EnvironmentExternalBase &env, 
                             bool rowNotColumns);
};


// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityHostInitGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    using GroupMerged::GroupMerged;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name, true);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateInitGroupMerged : public InitGroupMergedBase<GroupMerged<CustomUpdateInternal>,
                                                                           CustomUpdateVarAdapter>
{
public:
    using InitGroupMergedBase::InitGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};


// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomWUUpdateInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateInitGroupMerged : public InitGroupMergedBase<GroupMerged<CustomUpdateWUInternal>, 
                                                                             CustomUpdateVarAdapter>
{
public:
    using InitGroupMergedBase::InitGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomWUUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateSparseInitGroupMerged : public InitGroupMergedBase<GroupMerged<CustomUpdateWUInternal>, 
                                                                                   CustomUpdateVarAdapter>
{
public:
    using InitGroupMergedBase::InitGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomConnectivityUpdatePreInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityUpdatePreInitGroupMerged : public InitGroupMergedBase<GroupMerged<CustomConnectivityUpdateInternal>, 
                                                                                          CustomConnectivityUpdatePreVarAdapter>
{
public:
    using InitGroupMergedBase::InitGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomConnectivityUpdatePostInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityUpdatePostInitGroupMerged : public InitGroupMergedBase<GroupMerged<CustomConnectivityUpdateInternal>, 
                                                                                           CustomConnectivityUpdatePostVarAdapter>
{
public:
    using InitGroupMergedBase::InitGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomConnectivityUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomConnectivityUpdateSparseInitGroupMerged : public InitGroupMergedBase<GroupMerged<CustomConnectivityUpdateInternal>, 
                                                                                             CustomConnectivityUpdateVarAdapter>
{
public:
    using InitGroupMergedBase::InitGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitions) const
    {
        generateRunnerBase(backend, definitions, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, unsigned int batchSize);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace GeNN::CodeGenerator
