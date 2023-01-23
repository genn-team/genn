#pragma once

// GeNN code generator includes
#include "code_generator/groupMerged.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
class GENN_EXPORT PresynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    PresynapticUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend, 
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, backend, SynapseGroupMergedBase::Role::PresynapticUpdate, 
                               groups.front().get().getWUModel()->getSimCode() + groups.front().get().getWUModel()->getEventCode() + groups.front().get().getWUModel()->getEventThresholdConditionCode(), groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::PresynapticUpdate);
    }

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateSpikeEventThreshold(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    void generateSpikeEventUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    void generateSpikeUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    void generateProceduralConnectivity(const BackendBase &backend, CodeStream &os, Substitutions &popSubs) const;
    void generateToeplitzConnectivity(const BackendBase &backend, CodeStream &os, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PostsynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    PostsynapticUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend, 
                                  const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, backend, SynapseGroupMergedBase::Role::PostsynapticUpdate, 
                               groups.front().get().getWUModel()->getLearnPostCode(), groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::PostsynapticUpdate);
    }

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateSynapseUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;
    
    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDynamicsGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseDynamicsGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend, 
                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, typeContext, backend, SynapseGroupMergedBase::Role::SynapseDynamics, 
                               groups.front().get().getWUModel()->getSynapseDynamicsCode(), groups)
    {}

    boost::uuids::detail::sha1::digest_type getHashDigest() const
    {
        return SynapseGroupMergedBase::getHashDigest(SynapseGroupMergedBase::Role::SynapseDynamics);
    }

    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void generateSynapseUpdate(const BackendBase &backend, CodeStream &os, const ModelSpecMerged &modelMerged, Substitutions &popSubs) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDendriticDelayUpdateGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseDendriticDelayUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                           const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &group);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend,
                        CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                        CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                        CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace GeNN::CodeGenerator
