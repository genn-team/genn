#pragma once

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::InitGroupMergedBase
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
template<typename B, typename A>
class GENN_EXPORT InitGroupMergedBase : public B
{
public:
    using B::B;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Should the var init parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
    {
        return this->isParamValueHeterogeneous(paramName, 
                                               [&varName](const auto &g)
                                               { 
                                                   return A(g).getInitialisers().at(varName).getParams(); 
                                               });
    }

    //! Should the var init derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
    {
        return this->isParamValueHeterogeneous(paramName, 
                                               [&varName](const auto &g) 
                                               { 
                                                   return A(g).getInitialisers().at(varName).getDerivedParams();
                                               });
    }
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
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);
    
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const
        {
            updateBaseHash(hash);
            Utils::updateHash(getArchetype().getInitHashDigest(), hash);

        }
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
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);
        
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const
        {
            updateBaseHash(hash);
            Utils::updateHash(getArchetype().getPSInitHashDigest(), hash);
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
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);
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
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const
        {
            updateBaseHash(hash);
            Utils::updateHash(getArchetype().getWUPostInitHashDigest(), hash);
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
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);
        
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const
        {
            updateBaseHash(hash);
            Utils::updateHash(getArchetype().getWUPreInitHashDigest(), hash);
        }
    };

    NeuronInitGroupMerged(size_t index, const Type::TypeContext &typeContext,
                          const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
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

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

    const std::vector<CurrentSource> &getMergedCurrentSourceGroups() const { return m_MergedCurrentSourceGroups; }
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
    // Private methods
    //------------------------------------------------------------------------
    void genInitSpikeCount(const BackendBase &backend, EnvironmentExternalBase &env, bool spikeEvent, unsigned int batchSize);

    void genInitSpikes(const BackendBase &backend, EnvironmentExternalBase &env, bool spikeEvent, unsigned int batchSize);

    void genInitSpikeTime(const BackendBase &backend, EnvironmentExternalBase &env, const std::string &varName, unsigned int batchSize);
 
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<CurrentSource> m_MergedCurrentSourceGroups;
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

    void generateRunner(const BackendBase &backend, 
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

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

    void generateRunner(const BackendBase &backend, 
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

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

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateSparseRowInit(const BackendBase &backend, EnvironmentExternalBase &env);
    void generateSparseColumnInit(const BackendBase &backend, EnvironmentExternalBase &env);
    void generateKernelInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //! Should the sparse connectivity initialization parameter be implemented heterogeneously?
    bool isSparseConnectivityInitParamHeterogeneous(const std::string &paramName) const;

    //! Should the sparse connectivity initialization parameter be implemented heterogeneously?
    bool isSparseConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const;

    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    //! Generate either row or column connectivity init code
    void genInitConnectivity(const BackendBase &backend, EnvironmentExternalBase &env, bool rowNotColumns);
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
    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    //! Should the connectivity initialization parameter be implemented heterogeneously for EGP init?
    bool isConnectivityInitParamHeterogeneous(const std::string &paramName) const;

    //! Should the connectivity initialization derived parameter be implemented heterogeneously for EGP init?
    bool isConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const;
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

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

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

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

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

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

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
    InitGroupMergedBase::InitGroupMergedBase;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

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

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

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

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateInit(const BackendBase &backend, EnvironmentExternalBase &env, const ModelSpecMerged &modelMerged);

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace GeNN::CodeGenerator
