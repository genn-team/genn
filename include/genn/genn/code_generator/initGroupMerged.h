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
    class CurrentSource : public GroupMerged<CurrentSourceInternal>
    {
    public:
        CurrentSource(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                      const std::vector<std::reference_wrapper<const CurrentSourceInternal>> &groups);

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

    private:
        //----------------------------------------------------------------------------
        // Private methods
        //----------------------------------------------------------------------------
        //! Is the var init parameter referenced?
        bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;

        //! Should the var init parameter be implemented heterogeneously?
        bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

        //! Should the var init derived parameter be implemented heterogeneously?
        bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::InSynPSM
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
        void generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
        
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

    private:
        //----------------------------------------------------------------------------
        // Private methods
        //----------------------------------------------------------------------------
        //! Is the var init parameter referenced?
        bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;

        //! Should the var init parameter be implemented heterogeneously?
        bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

        //! Should the var init derived parameter be implemented heterogeneously?
        bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynPreOutput
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
        void generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::InSynWUMPostCode
    //----------------------------------------------------------------------------
    //! Child group merged for incoming synapse groups with postsynaptic variables
    class InSynWUMPostVars : public GroupMerged<SynapseGroupInternal>
    {
    public:
        InSynWUMPostVars(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                         const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

    private:
        //----------------------------------------------------------------------------
        // Private methods
        //----------------------------------------------------------------------------
        //! Is the var init parameter referenced?
        bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;

        //! Should the var init parameter be implemented heterogeneously?
        bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

        //! Should the var init derived parameter be implemented heterogeneously?
        bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;
    };

    //----------------------------------------------------------------------------
    // GeNN::CodeGenerator::NeuronInitGroupMerged::OutSynWUMPreVars
    //----------------------------------------------------------------------------
    //! Child group merged for outgoing synapse groups with presynaptic variables
    class OutSynWUMPreVars: public GroupMerged<SynapseGroupInternal>
    {
    public:
        OutSynWUMPreVars(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                         const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        void generate(const BackendBase &backend, CodeStream &os, const NeuronInitGroupMerged &ng,
                      const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
        
        //! Update hash with child groups
        void updateHash(boost::uuids::detail::sha1 &hash) const;

    private:
        //----------------------------------------------------------------------------
        // Private methods
        //----------------------------------------------------------------------------
        //! Is the var init parameter referenced?
        bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;

        //! Should the var init parameter be implemented heterogeneously?
        bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

        //! Should the var init derived parameter be implemented heterogeneously?
        bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;
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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void genInitSpikeCount(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                           bool spikeEvent, unsigned int batchSize) const;

    void genInitSpikes(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                       bool spikeEvent, unsigned int batchSize) const;

    void genInitSpikeTime(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                          const std::string &varName, unsigned int batchSize) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<CurrentSource> m_CurrentSources;
    std::vector<InSynPSM> m_InSynPSMs;
    std::vector<OutSynPreOutput> m_OutSynPreOutput;
    std::vector<CurrentSource> m_SortedInSynWithPostCode;
    std::vector<InSynWUMPostVars> m_InSynWUMPostVars;
    std::vector<OutSynWUMPreVars> m_OutSynWUMPreVars;
};


//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend, 
                                const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, backend, SynapseGroupMergedBase::Role::Init, "", groups)
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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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
    SynapseSparseInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend, 
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, backend, SynapseGroupMergedBase::Role::SparseInit, "", groups)
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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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
    SynapseConnectivityInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, backend, SynapseGroupMergedBase::Role::ConnectivityInit, "", groups)
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

    void generateSparseRowInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    void generateSparseColumnInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    void generateKernelInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    //! Generate either row or column connectivity init code
    void genInitConnectivity(CodeStream &os, Substitutions &popSubs, bool rowNotColumns) const;
};


// ----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityHostInitGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseConnectivityHostInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                           const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged) const;

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
protected:
    CustomUpdateInitGroupMergedBase(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                    const std::vector<std::reference_wrapper<const G>> &groups)
    :   GroupMerged<G>(index, typeContext, groups)
    {
        // Loop through variables
        A archetypeAdaptor(this->getArchetype());
        for (const auto &var : archetypeAdaptor.getDefs()) {
            // If we're not initialising or if there is initialization code for this variable
            const auto &varInit = archetypeAdaptor.getInitialisers().at(var.name);
            if (!varInit.getSnippet()->getCode().empty()) {
                this->addPointerField(var.type, var.name, backend.getDeviceVarPrefix() + var.name);
            }

            // Add any var init EGPs to structure
            this->addEGPs(varInit.getSnippet()->getExtraGlobalParams(), backend.getDeviceVarPrefix(), var.name);
        }

        this->template addHeterogeneousVarInitParams<CustomUpdateInitGroupMergedBase<G, A>, A>(
            &CustomUpdateInitGroupMergedBase<G, A>::isVarInitParamHeterogeneous);

        this->template addHeterogeneousVarInitDerivedParams<CustomUpdateInitGroupMergedBase<G, A>, A>(
            &CustomUpdateInitGroupMergedBase<G, A>::isVarInitDerivedParamHeterogeneous);
    }

    //----------------------------------------------------------------------------
    // Protected methods
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
    CustomUpdateInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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
    void genKernelIndex(std::ostream &os, const CodeGenerator::Substitutions &subs) const
    {
        return CodeGenerator::genKernelIndex(this, os, subs, getGroupKernelSize);
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
    CustomWUUpdateSparseInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

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
    CustomConnectivityUpdateSparseInitGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
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

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace GeNN::CodeGenerator
