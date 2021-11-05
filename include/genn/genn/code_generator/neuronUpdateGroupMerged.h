#pragma once

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
namespace CodeGenerator
{
class GENN_EXPORT NeuronUpdateGroupMerged : public NeuronGroupMergedBase
{
public:
    NeuronUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Should the incoming synapse weight update model parameter be implemented heterogeneously?
    bool isInSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the incoming synapse weight update model derived parameter be implemented heterogeneously?
    bool isInSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model parameter be implemented heterogeneously?
    bool isOutSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model derived parameter be implemented heterogeneously?
    bool isOutSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Get sorted vectors of incoming synapse groups with postsynaptic code belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeInSynWithPostCode() const { return m_SortedInSynWithPostCode.front(); }

    //! Get sorted vectors of outgoing synapse groups with presynaptic code belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeOutSynWithPreCode() const { return m_SortedOutSynWithPreCode.front(); }

    //! Get hash digest used for detecting changes
    boost::uuids::detail::sha1::digest_type getHashDigest() const;

    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }
    
    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::string getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    static std::string getReadVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    static std::string getWriteVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);

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
                       bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, size_t) const,
                       bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t) const,
                       const std::string&(SynapseGroupInternal::*getFusedVarSuffix)(void) const);

    //! Is the incoming synapse weight update model parameter referenced?
    bool isInSynWUMParamReferenced(size_t childIndex, size_t paramIndex) const;

    //! Is the incoming synapse weight update model derived parameter referenced?
    bool isInSynWUMDerivedParamReferenced(size_t childIndex, size_t paramIndex) const;

    //! Is the outgoing synapse weight update model parameter referenced?
    bool isOutSynWUMParamReferenced(size_t childIndex, size_t paramIndex) const;

    //! Is the outgoing synapse weight update model derived parameter referenced?
    bool isOutSynWUMDerivedParamReferenced(size_t childIndex, size_t paramIndex) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedInSynWithPostCode;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedOutSynWithPreCode;
};
}   // namespace CodeGenerator
