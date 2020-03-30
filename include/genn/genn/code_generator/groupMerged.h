#pragma once

// Standard includes
#include <functional>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

// Forward declarations
namespace CodeGenerator
{
class BackendBase;
class CodeStream;
class MergedStructData;

template<typename T>
class MergedStructGenerator;
}

//----------------------------------------------------------------------------
// CodeGenerator::GroupMerged
//----------------------------------------------------------------------------
//! Very thin wrapper around a number of groups which have been merged together
namespace CodeGenerator
{
template<typename G>
class GroupMerged
{
public:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef G GroupInternal;

    GroupMerged(size_t index, const std::vector<std::reference_wrapper<const GroupInternal>> groups)
    :   m_Index(index), m_Groups(std::move(groups))
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getIndex() const { return m_Index; }

    //! Get 'archetype' neuron group - it's properties represent those of all other merged neuron groups
    const GroupInternal &getArchetype() const { return m_Groups.front().get(); }

    //! Gets access to underlying vector of neuron groups which have been merged
    const std::vector<std::reference_wrapper<const GroupInternal>> &getGroups() const{ return m_Groups; }

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    //! Helper to test whether parameter values are heterogeneous within merged group
    template<typename P>
    bool isParamValueHeterogeneous(size_t index, P getParamValuesFn) const
    {
        // Get value of parameter in archetype group
        const double archetypeValue = getParamValuesFn(getArchetype()).at(index);

        // Return true if any parameter values differ from the archetype value
        return std::any_of(getGroups().cbegin(), getGroups().cend(),
                           [archetypeValue, index, getParamValuesFn](const GroupInternal &g)
                           {
                               return (getParamValuesFn(g).at(index) != archetypeValue);
                           });
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const size_t m_Index;
    std::vector<std::reference_wrapper<const GroupInternal>> m_Groups;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronSpikeQueueUpdateMergedGroup
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronSpikeQueueUpdateMergedGroup : public GroupMerged<NeuronGroupInternal>
{
public:
    NeuronSpikeQueueUpdateMergedGroup(size_t index, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision) const;

    void genMergedGroupSpikeCountReset(CodeStream &os) const;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronGroupMergedBase : public GroupMerged<NeuronGroupInternal>
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Should the parameter be implemented heterogeneously?
    bool isParamHeterogeneous(size_t index) const;

    //! Should the derived parameter be implemented heterogeneously?
    bool isDerivedParamHeterogeneous(size_t index) const;

    //! Should the current source parameter be implemented heterogeneously?
    bool isCurrentSourceParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the current source derived parameter be implemented heterogeneously?
    bool isCurrentSourceDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    NeuronGroupMergedBase(size_t index, bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    const std::vector<std::vector<std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>>>> &getSortedMergedInSyns() const { return m_SortedMergedInSyns; }
    const std::vector<std::vector<CurrentSourceInternal *>> &getSortedCurrentSources() const { return m_SortedCurrentSources; }
    const std::vector<std::vector<SynapseGroupInternal *>> &getSortedInSynWithPostCode() const { return m_SortedInSynWithPostCode; }
    const std::vector<std::vector<SynapseGroupInternal *>> &getSortedOutSynWithPreCode() const { return m_SortedOutSynWithPreCode; }

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
               CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
               CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
               MergedStructData &mergedStructData, const std::string &precision,
               const std::string &timePrecision, bool init) const;
private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    template<typename T, typename G, typename C>
    void orderNeuronGroupChildren(const std::vector<T> &archetypeChildren,
                                  std::vector<std::vector<T>> &sortedGroupChildren,
                                  G getVectorFunc, C isCompatibleFunc) const
    {
        // Reserve vector of vectors to hold children for all neuron groups, in archetype order
        sortedGroupChildren.reserve(archetypeChildren.size());

        // Loop through groups
        for(const auto &g : getGroups()) {
            // Make temporary copy of this group's children
            std::vector<T> tempChildren((g.get().*getVectorFunc)());

            assert(tempChildren.size() == archetypeChildren.size());

            // Reserve vector for this group's children
            sortedGroupChildren.emplace_back();
            sortedGroupChildren.back().reserve(tempChildren.size());

            // Loop through archetype group's children
            for(const auto &archetypeG : archetypeChildren) {
                // Find compatible child in temporary list
                const auto otherChild = std::find_if(tempChildren.cbegin(), tempChildren.cend(),
                                                     [archetypeG, isCompatibleFunc](const T &g)
                                                     {
                                                         return isCompatibleFunc(archetypeG, g);
                                                     });
                assert(otherChild != tempChildren.cend());

                // Add pointer to vector of compatible merged in syns
                sortedGroupChildren.back().push_back(*otherChild);

                // Remove from original vector
                tempChildren.erase(otherChild);
            }
        }
    }
    
    template<typename T, typename G, typename C>
    void orderNeuronGroupChildren(std::vector<std::vector<T>> &sortedGroupChildren,
                                  G getVectorFunc, C isCompatibleFunc) const
    {
        const std::vector<T> &archetypeChildren = (getArchetype().*getVectorFunc)();
        orderNeuronGroupChildren(archetypeChildren, sortedGroupChildren, getVectorFunc, isCompatibleFunc);
    }
    
    template<typename T, typename G>
    bool isChildParamValueHeterogeneous(size_t childIndex, size_t paramIndex, const std::vector<std::vector<T>> &sortedGroupChildren,
                                        G getParamValuesFn) const
    {
        // Get value of archetype derived parameter
        const double firstValue = getParamValuesFn(sortedGroupChildren[0][childIndex]).at(paramIndex);

        // Loop through groups within merged group
        for(size_t i = 0; i < sortedGroupChildren.size(); i++) {
            const auto group = sortedGroupChildren[i][childIndex];
            if(getParamValuesFn(group).at(paramIndex) != firstValue) {
                return true;
            }
        }
        return false;
    }

    void addMergedInSynPointerField(MergedStructGenerator<NeuronGroupMergedBase> &gen,
                                    const std::string &type, const std::string &name, 
                                    size_t archetypeIndex, const std::string &prefix) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>>>> m_SortedMergedInSyns;
    std::vector<std::vector<CurrentSourceInternal*>> m_SortedCurrentSources;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedInSynWithPostCode;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedOutSynWithPreCode;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronUpdateGroupMerged : public NeuronGroupMergedBase
{
public:
    NeuronUpdateGroupMerged(size_t index, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get the expression to calculate the queue offset for accessing state of variables this timestep
    std::string getCurrentQueueOffset() const;

    //! Get the expression to calculate the queue offset for accessing state of variables in previous timestep
    std::string getPrevQueueOffset() const;

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronInitGroupMerged : public NeuronGroupMergedBase
{
public:
    NeuronInitGroupMerged(size_t index, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const;
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateMergedGroup
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDendriticDelayUpdateMergedGroup : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseDendriticDelayUpdateMergedGroup(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision) const;
};

// ----------------------------------------------------------------------------
// SynapseConnectivityHostInitMergedGroup
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityHostInitMergedGroup : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseConnectivityHostInitMergedGroup(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision) const;

    //! Should the connectivity initialization parameter be implemented heterogeneously for EGP init?
    bool isConnectivityHostInitParamHeterogeneous(size_t paramIndex) const;

    //! Should the connectivity initialization derived parameter be implemented heterogeneously for EGP init?
    bool isConnectivityHostInitDerivedParamHeterogeneous(size_t paramIndex) const;
};

// ----------------------------------------------------------------------------
// SynapseConnectivityInitMergedGroup
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityInitMergedGroup : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseConnectivityInitMergedGroup(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision) const;
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseGroupMergedBase : public GroupMerged<SynapseGroupInternal>
{
public:
   
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get the expression to calculate the delay slot for accessing
    //! Presynaptic neuron state variables, taking into account axonal delay
    std::string getPresynapticAxonalDelaySlot() const;

    //! Get the expression to calculate the delay slot for accessing
    //! Postsynaptic neuron state variables, taking into account back propagation delay
    std::string getPostsynapticBackPropDelaySlot() const;

    std::string getDendriticDelayOffset(const std::string &offset = "") const;

    //! Should the weight update model variable initialization parameter be implemented heterogeneously?
    bool isWUVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const;
    
    //! Should the weight update model variable initialization derived parameter be implemented heterogeneously?
    bool isWUVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const;

protected:
    //----------------------------------------------------------------------------
    // Enumerations
    //----------------------------------------------------------------------------
    enum class Role
    {
        PresynapticUpdate,
        PostsynapticUpdate,
        SynapseDynamics,
        DenseInit,
        SparseInit,
    };

    SynapseGroupMergedBase(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
        : GroupMerged<SynapseGroupInternal>(index, groups)
    {}

    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision, const std::string &name, Role role) const;
private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void addPSPointerField(MergedStructGenerator<SynapseGroupMergedBase> &gen,
                           const std::string &type, const std::string &name, const std::string &prefix) const;
    void addSrcPointerField(MergedStructGenerator<SynapseGroupMergedBase> &gen,
                            const std::string &type, const std::string &name, const std::string &prefix) const;
    void addTrgPointerField(MergedStructGenerator<SynapseGroupMergedBase> &gen,
                            const std::string &type, const std::string &name, const std::string &prefix) const;
};

//----------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PresynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    PresynapticUpdateGroupMerged(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, groups)
    {}

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const
    {
        SynapseGroupMergedBase::generate(backend, definitionsInternal, definitionsInternalFunc,
                                         definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc, 
                                         mergedStructData, precision, timePrecision, 
                                         "PresynapticUpdate", SynapseGroupMergedBase::Role::PresynapticUpdate);
    }
};

//----------------------------------------------------------------------------
// CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PostsynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    PostsynapticUpdateGroupMerged(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, groups)
    {}

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const
    {
        SynapseGroupMergedBase::generate(backend, definitionsInternal, definitionsInternalFunc,
                                         definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                                         mergedStructData, precision, timePrecision,
                                         "PostsynapticUpdate", SynapseGroupMergedBase::Role::PostsynapticUpdate);
    }
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDynamicsGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseDynamicsGroupMerged(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, groups)
    {}

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const
    {
        SynapseGroupMergedBase::generate(backend, definitionsInternal, definitionsInternalFunc,
                                         definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                                         mergedStructData, precision, timePrecision,
                                         "SynapseDynamics", SynapseGroupMergedBase::Role::SynapseDynamics);
    }
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDenseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDenseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseDenseInitGroupMerged(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, groups)
    {}

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const
    {
        SynapseGroupMergedBase::generate(backend, definitionsInternal, definitionsInternalFunc,
                                         definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                                         mergedStructData, precision, timePrecision,
                                         "SynapseDenseInit", SynapseGroupMergedBase::Role::DenseInit);
    }
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseSparseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseSparseInitGroupMerged(size_t index, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, groups)
    {}

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const
    {
        SynapseGroupMergedBase::generate(backend, definitionsInternal, definitionsInternalFunc,
                                         definitionsInternalVar, runnerVarDecl, runnerMergedStructAlloc,
                                         mergedStructData, precision, timePrecision,
                                         "SynapseSparseInit", SynapseGroupMergedBase::Role::SparseInit);
    }
};
}   // namespace CodeGenerator
