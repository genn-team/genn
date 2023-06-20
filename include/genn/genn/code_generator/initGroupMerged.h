#pragma once

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT NeuronInitGroupMerged : public NeuronGroupMergedBase
{
public:
    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::CurrentSource
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
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);
    
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the var init parameter be implemented heterogeneously?
        bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

        //! Should the var init derived parameter be implemented heterogeneously?
        bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

    private:
        //----------------------------------------------------------------------------
        // Private methods
        //----------------------------------------------------------------------------
        //! Is the var init parameter referenced?
        bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::InSynPSM
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
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);
        
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the var init parameter be implemented heterogeneously?
        bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

        //! Should the var init derived parameter be implemented heterogeneously?
        bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

    private:
        //----------------------------------------------------------------------------
        // Private methods
        //----------------------------------------------------------------------------
        //! Is the var init parameter referenced?
        bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;
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
    class InSynWUMPostVars : public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the var init parameter be implemented heterogeneously?
        bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

        //! Should the var init derived parameter be implemented heterogeneously?
        bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

    private:
        //----------------------------------------------------------------------------
        // Private methods
        //----------------------------------------------------------------------------
        //! Is the var init parameter referenced?
        bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynWUMPreVars
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with presynaptic variables
    class OutSynWUMPreVars: public ChildGroupMerged<SynapseGroupInternal>
    {
    public:
        using ChildGroupMerged::ChildGroupMerged;

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, EnvironmentExternalBase &env, 
                      NeuronInitGroupMerged &ng, const ModelSpecMerged &modelMerged);
        
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

        //! Should the var init parameter be implemented heterogeneously?
        bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

        //! Should the var init derived parameter be implemented heterogeneously?
        bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

    private:
        //----------------------------------------------------------------------------
        // Private methods
        //----------------------------------------------------------------------------
        //! Is the var init parameter referenced?
        bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;  
    };

    NeuronInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
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

    //! Should the var init parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

    //! Should the var init derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

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
class GENN_EXPORT SynapseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseInitGroupMerged(size_t index, const Type::TypeContext &typeContext, 
                           const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, SynapseGroupMergedBase::Role::Init, "", groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::Init);
    }

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
class GENN_EXPORT SynapseSparseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseSparseInitGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, SynapseGroupMergedBase::Role::SparseInit, "", groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::SparseInit);
    }

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
class GENN_EXPORT SynapseConnectivityInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseConnectivityInitGroupMerged(size_t index, const Type::TypeContext &typeContext,
                                       const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, SynapseGroupMergedBase::Role::ConnectivityInit, "", groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::ConnectivityInit);
    }

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

     //! Is the connectivity initialization parameter referenced?
    bool isSparseConnectivityInitParamReferenced(const std::string &paramName) const;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateInitGroupMergedBase
//----------------------------------------------------------------------------
//! Boilerplate base class for creating merged init groups for various types of CustomUpdate CustomConectivityUpdate
template<typename G, typename A>
class CustomUpdateInitGroupMergedBase : public GroupMerged<G>
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Should the var init parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
    {
        return (isVarInitParamReferenced(varName, paramName) &&
                this->isParamValueHeterogeneous(paramName, 
                                                [&varName](const G &cg)
                                                { 
                                                    A archetypeAdaptor(cg);
                                                    return archetypeAdaptor.getInitialisers().at(varName).getParams(); 
                                                }));
    }

    //! Should the var init derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
    {
        return (isVarInitParamReferenced(varName, paramName) &&
                this->isParamValueHeterogeneous(paramName, 
                                                [&varName](const G &cg) 
                                                { 
                                                    A archetypeAdaptor(cg);
                                                    return archetypeAdaptor.getInitialisers().at(varName).getDerivedParams();
                                                }));
    }
protected:
    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    void updateBaseHash(boost::uuids::detail::sha1 &hash) const
    {
        // Update hash with archetype's hash digest
        Utils::updateHash(this->getArchetype().getInitHashDigest(), hash);

        // Update hash with each group's variable initialisation parameters and derived parameters
        this->template updateVarInitParamHash<CustomUpdateInitGroupMergedBase<G, A>, A>(
            &CustomUpdateInitGroupMergedBase<G, A>::isVarInitParamHeterogeneous, hash);

        this->template updateVarInitDerivedParamHash<CustomUpdateInitGroupMergedBase<G, A>, A>(
            &CustomUpdateInitGroupMergedBase<G, A>::isVarInitDerivedParamHeterogeneous, hash);
    }

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    //! Is the var init parameter referenced?
    bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
    {
        A archetypeAdaptor(this->getArchetype());
        const auto *varInitSnippet = archetypeAdaptor.getInitialisers().at(varName).getSnippet();
        return this->isParamReferenced({varInitSnippet->getCode()}, paramName);
    }

    //! Is the var init derived parameter referenced?
    bool isVarInitDerivedParamReferenced(const std::string &varName, const std::string &paramName) const
    {
        A archetypeAdaptor(this->getArchetype());
        const auto *varInitSnippet = archetypeAdaptor.getInitialisers().at(varName).getSnippet();
        return this->isParamReferenced({varInitSnippet->getCode()}, paramName);
    }
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomUpdateInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateInternal, CustomUpdateVarAdapter>
{
public:
    using CustomUpdateInitGroupMergedBase::CustomUpdateInitGroupMergedBase;

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
class GENN_EXPORT CustomWUUpdateInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal,
                                                                                         CustomUpdateVarAdapter>
{
public:
    CustomWUUpdateInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                  const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

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
    std::string getKernelIndex(EnvironmentExternalBase &env) const
    {
        return CodeGenerator::getKernelIndex(this, env, getGroupKernelSize);
    }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //----------------------------------------------------------------------------
    // Private static methods
    //----------------------------------------------------------------------------
    static const std::vector<unsigned int> &getGroupKernelSize(const CustomUpdateWUInternal &g)
    {
        return g.getSynapseGroup()->getKernelSize();
    }
};

// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::CustomWUUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateSparseInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal,
                                                                                               CustomUpdateVarAdapter>
{
public:
    using CustomUpdateInitGroupMergedBase::CustomUpdateInitGroupMergedBase;

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
class GENN_EXPORT CustomConnectivityUpdatePreInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal,
                                                                                                      CustomConnectivityUpdatePreVarAdapter>
{
public:
    CustomConnectivityUpdatePreInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                               const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups);

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
class GENN_EXPORT CustomConnectivityUpdatePostInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal,
                                                                                                       CustomConnectivityUpdatePostVarAdapter>
{
public:
    CustomConnectivityUpdatePostInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                                const std::vector<std::reference_wrapper<const CustomConnectivityUpdateInternal>> &groups);

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
class GENN_EXPORT CustomConnectivityUpdateSparseInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomConnectivityUpdateInternal,
                                                                                                         CustomConnectivityUpdateVarAdapter>
{
public:
    using CustomUpdateInitGroupMergedBase::CustomUpdateInitGroupMergedBase;

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
