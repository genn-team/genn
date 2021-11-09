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
    //! Get hash digest used for detecting changes
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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
    //! Helper to generate merged struct fields for WU pre and post vars
    void generateWUVar(const BackendBase &backend, const std::string &fieldPrefixStem,
                       const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                       Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                       const std::vector<Models::VarInit>&(SynapseGroupInternal::*getVarInitialiserFn)(void) const,
                       bool(NeuronInitGroupMerged::*isParamHeterogeneousFn)(size_t, size_t, size_t) const,
                       bool(NeuronInitGroupMerged::*isDerivedParamHeterogeneousFn)(size_t, size_t, size_t) const,
                       const std::string&(SynapseGroupInternal::*getFusedVarSuffix)(void) const);

    //! Should the incoming synapse weight update model var init parameter be implemented heterogeneously?
    bool isInSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the incoming synapse weight update model var init derived parameter be implemented heterogeneously?
    bool isInSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model var init parameter be implemented heterogeneously?
    bool isOutSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model var init derived parameter be implemented heterogeneously?
    bool isOutSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Get sorted vectors of incoming synapse groups with postsynaptic variables belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeInSynWithPostVars() const { return m_SortedInSynWithPostVars.front(); }

    //! Get sorted vectors of outgoing synapse groups with presynaptic variables belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeOutSynWithPreVars() const { return m_SortedOutSynWithPreVars.front(); }

    //! Is the incoming synapse weight update model var init parameter referenced?
    bool isInSynWUMVarInitParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Is the incoming synapse weight update model var init derived parameter referenced?
    bool isInSynWUMVarInitDerivedParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Is the outgoing synapse weight update model var init parameter referenced?
    bool isOutSynWUMVarInitParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Is the outgoing synapse weight update model var init derived parameter referenced?
    bool isOutSynWUMVarInitDerivedParamReferenced(size_t childIndex, size_t varIndex, size_t paramIndex) const;

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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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
    void genInitConnectivity(CodeStream &os, Substitutions &popSubs, const std::string &ftype, bool rowNotColumns) const;
};

//----------------------------------------------------------------------------
// CodeGenerator::CustomUpdateInitGroupMergedBase
//----------------------------------------------------------------------------
template<typename G>
class CustomUpdateInitGroupMergedBase : public GroupMerged<G>
{
protected:
    CustomUpdateInitGroupMergedBase(size_t index, const std::string &precision, const BackendBase &backend,
                                    const std::vector<std::reference_wrapper<const G>> &groups)
    :   GroupMerged<G>(index, precision, groups)
    {
         // Loop through variables
        const CustomUpdateModels::Base *cm = this->getArchetype().getCustomUpdateModel();
        const auto vars = cm->getVars();
        const auto &varInit = this->getArchetype().getVarInitialisers();
        assert(vars.size() == varInit.size());
        for(size_t v = 0; v < vars.size(); v++) {
            // If we're not initialising or if there is initialization code for this variable
            const auto var = vars[v];
            if(!varInit[v].getSnippet()->getCode().empty()) {
                this->addPointerField(var.type, var.name, backend.getDeviceVarPrefix() + var.name);
            }

            // Add any var init EGPs to structure
            this->addEGPs(varInit[v].getSnippet()->getExtraGlobalParams(), backend.getDeviceVarPrefix(), var.name);
        }

        this->template addHeterogeneousVarInitParams<CustomUpdateInitGroupMergedBase<G>>(
            vars, &G::getVarInitialisers,
            &CustomUpdateInitGroupMergedBase<G>::isVarInitParamHeterogeneous);

        this->template addHeterogeneousVarInitDerivedParams<CustomUpdateInitGroupMergedBase<G>>(
            vars, &G::getVarInitialisers,
            &CustomUpdateInitGroupMergedBase<G>::isVarInitDerivedParamHeterogeneous);
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

    //! Is the var init parameter referenced?
    bool isVarInitParamReferenced(size_t varIndex, size_t paramIndex) const
    {
        // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
        const auto *varInitSnippet = this->getArchetype().getVarInitialisers().at(varIndex).getSnippet();
        const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
        return this->isParamReferenced({varInitSnippet->getCode()}, paramName);
    }

    //! Is the var init derived parameter referenced?
    bool isVarInitDerivedParamReferenced(size_t varIndex, size_t paramIndex) const
    {
        // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
        const auto *varInitSnippet = this->getArchetype().getVarInitialisers().at(varIndex).getSnippet();
        const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
        return this->isParamReferenced({varInitSnippet->getCode()}, derivedParamName);
    }

    //! Should the var init parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const
    {
        return (isVarInitParamReferenced(varIndex, paramIndex) &&
                this->isParamValueHeterogeneous(paramIndex, [varIndex](const G &cg) { return cg.getVarInitialisers().at(varIndex).getParams(); }));
    }

    //! Should the var init derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const
    {
        return (isVarInitDerivedParamReferenced(varIndex, paramIndex) &&
                this->isParamValueHeterogeneous(paramIndex, [varIndex](const G &cg) { return cg.getVarInitialisers().at(varIndex).getDerivedParams(); }));
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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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

}   // namespace CodeGenerator
