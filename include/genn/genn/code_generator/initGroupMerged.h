#pragma once

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
namespace CodeGenerator
{
class GENN_EXPORT NeuronInitGroupMerged : public NeuronGroupMergedBase
{
public:
    NeuronInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                          const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Should the incoming synapse weight update model var init parameter be implemented heterogeneously?
    bool isInSynWUMVarInitParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Should the incoming synapse weight update model var init derived parameter be implemented heterogeneously?
    bool isInSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Should the outgoing synapse weight update model var init parameter be implemented heterogeneously?
    bool isOutSynWUMVarInitParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Should the outgoing synapse weight update model var init derived parameter be implemented heterogeneously?
    bool isOutSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Get sorted vectors of incoming synapse groups with postsynaptic variables belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeInSynWithPostVars() const { return m_SortedInSynWithPostVars.front(); }

    //! Get sorted vectors of outgoing synapse groups with presynaptic variables belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeOutSynWithPreVars() const { return m_SortedOutSynWithPreVars.front(); }

    //! Get hash digest used for detecting changes
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
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
    //! Helper to generate merged struct fields for WU pre and post vars
    void generateWUVar(const BackendBase &backend, const std::string &fieldPrefixStem,
                       const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                       Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                       const std::unordered_map<std::string, Models::VarInit>&(SynapseGroupInternal::*getVarInitialiserFn)(void) const,
                       bool(NeuronInitGroupMerged::*isParamHeterogeneousFn)(size_t, const std::string&, const std::string&) const,
                       bool(NeuronInitGroupMerged::*isDerivedParamHeterogeneousFn)(size_t, const std::string&, const std::string&) const);

    //! Is the incoming synapse weight update model var init parameter referenced?
    bool isInSynWUMVarInitParamReferenced(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Is the outgoing synapse weight update model var init parameter referenced?
    bool isOutSynWUMVarInitParamReferenced(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    void genInitSpikeCount(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                           bool spikeEvent, unsigned int batchSize) const;

    void genInitSpikes(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                       bool spikeEvent, unsigned int batchSize) const;

    void genInitSpikeTime(CodeStream &os, const BackendBase &backend, const Substitutions &popSubs,
                          const std::string &varName, unsigned int batchSize) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedInSynWithPostVars;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedOutSynWithPreVars;
};


//----------------------------------------------------------------------------
// CodeGenerator::SynapseDenseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDenseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseDenseInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::DenseInit, "", groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::DenseInit);
    }

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseSparseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseSparseInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::SparseInit, "", groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::SparseInit);
    }

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseKernelInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseKernelInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseKernelInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::KernelInit, "", groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::KernelInit);
    }

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};


// ----------------------------------------------------------------------------
// CodeGenerator::SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseConnectivityInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::ConnectivityInit, "", groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::ConnectivityInit);
    }

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
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
    void genInitConnectivity(CodeStream &os, Substitutions &popSubs, const std::string &ftype, bool rowNotColumns) const;
};

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateInitGroupMergedBase
//----------------------------------------------------------------------------
template<typename G>
class CustomUpdateInitGroupMergedBase : public RuntimeGroupMerged<G>
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    //! Should the var init parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const
    {
        return (isVarInitParamReferenced(varName, paramName) &&
                this->isParamValueHeterogeneous(paramName, [varName](const G &cg) { return cg.getVarInitialisers().at(varName).getParams(); }));
    }

    //! Should the var init derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const
    {
        return (isVarInitParamReferenced(varName, paramName) &&
                this->isParamValueHeterogeneous(paramName, [varName](const G &cg) { return cg.getVarInitialisers().at(varName).getDerivedParams(); }));
    }

protected:
    CustomUpdateInitGroupMergedBase(size_t index, const std::string &precision, const BackendBase &backend,
                                    const std::vector<std::reference_wrapper<const G>> &groups)
    :   RuntimeGroupMerged<G>(index, precision, backend, groups)
    {
         // Loop through variables
        const CustomUpdateModels::Base *cm = this->getArchetype().getCustomUpdateModel();
        for(const auto &var : cm->getVars()) {
            const auto &varInit = this->getArchetype().getVarInitialisers().at(var.name);

            // If we're not initialising or if there is initialization code for this variable
            if(!varInit.getSnippet()->getCode().empty()) {
                this->addPointerField(var.type, var.name, backend.getDeviceVarPrefix() + var.name);
            }

            // Add any var init EGPs to structure
            this->addEGPs(varInit.getSnippet()->getExtraGlobalParams(), var.name);
        }

        this->template addHeterogeneousVarInitParams<CustomUpdateInitGroupMergedBase<G>>(
            &G::getVarInitialisers, &CustomUpdateInitGroupMergedBase<G>::isVarInitParamHeterogeneous);

        this->template addHeterogeneousVarInitDerivedParams<CustomUpdateInitGroupMergedBase<G>>(
            &G::getVarInitialisers, &CustomUpdateInitGroupMergedBase<G>::isVarInitDerivedParamHeterogeneous);
    }

    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    void updateBaseHash(boost::uuids::detail::sha1 &hash) const
    {
        // Update hash with archetype's hash digest
        Utils::updateHash(this->getArchetype().getInitHashDigest(), hash);
        
        // Update hash with each group's variable initialisation parameters and derived parameters
        this->template updateVarInitParamHash<CustomUpdateInitGroupMergedBase<G>>(
            &G::getVarInitialisers, &CustomUpdateInitGroupMergedBase<G>::isVarInitParamHeterogeneous, hash);
        
        this->template updateVarInitDerivedParamHash<CustomUpdateInitGroupMergedBase<G>>(
            &G::getVarInitialisers, &CustomUpdateInitGroupMergedBase<G>::isVarInitDerivedParamHeterogeneous, hash);
    }

private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    //! Is the var init parameter referenced?
    bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const
    {
        const auto *varInitSnippet = this->getArchetype().getVarInitialisers().at(varName).getSnippet();
        return this->isParamReferenced({varInitSnippet->getCode()}, paramName);
    }

    //! Is the var init derived parameter referenced?
    bool isVarInitDerivedParamReferenced(const std::string &varName, const std::string &paramName) const
    {
        const auto *varInitSnippet = this->getArchetype().getVarInitialisers().at(varName).getSnippet();
        return this->isParamReferenced({varInitSnippet->getCode()}, paramName);
    }
};

// ----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateInternal>
{
public:
    CustomUpdateInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};


// ----------------------------------------------------------------------------
// CodeGenerator::CustomWUUpdateDenseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateDenseInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal>
{
public:
    CustomWUUpdateDenseInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// CodeGenerator::CustomWUUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateSparseInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal>
{
public:
    CustomWUUpdateSparseInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                        const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                        const MergedRunnerMap &mergedRunnerMap) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, mergedRunnerMap, name);
    }

    void generateInit(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

}   // namespace CodeGenerator
