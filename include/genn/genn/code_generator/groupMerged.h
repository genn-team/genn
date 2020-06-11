#pragma once

// Standard includes
#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "currentSourceInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

// GeNN code generator includes
#include "code_generator/mergedStructGenerator.h"

// Forward declarations
namespace CodeGenerator
{
class BackendBase;
class CodeStream;
class MergedStructData;
}

//----------------------------------------------------------------------------
// CodeGenerator::GroupMerged
//----------------------------------------------------------------------------
//! Very thin wrapper around a number of groups which have been merged together
namespace CodeGenerator
{
template<typename G, typename M>
class GroupMerged
{
public:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef G GroupInternal;

    GroupMerged(size_t index, const std::string &precision, const std::vector<std::reference_wrapper<const GroupInternal>> groups)
    :   m_Gen(*static_cast<M*>(this), precision), m_Index(index), m_Groups(std::move(groups))
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getIndex() const { return m_Index; }

    //! Get 'archetype' neuron group - it's properties represent those of all other merged neuron groups
    const GroupInternal &getArchetype() const { return m_Groups.front().get(); }

    //! Gets access to underlying vector of neuron groups which have been merged
    const std::vector<std::reference_wrapper<const GroupInternal>> &getGroups() const{ return m_Groups; }

    //! Generate suitable struct to hold this merged group
    void generateStruct(const BackendBase &backend, CodeStream &os,
                        const std::string &name, const std::string &prefix = "") const
    {
        m_Gen.generateStruct(backend, os, name, prefix);

    }

    size_t getStructArraySize(const BackendBase &backend) const
    {
        return m_Gen.getArraySize(backend);
    }

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

    //! Helper to test whether parameter values are heterogeneous within merged group
    template<typename P>
    bool isParamValueHeterogeneous(std::initializer_list<std::string> codeStrings, const std::string &paramName,
                                   size_t index, P getParamValuesFn) const
    {
        // If none of the code strings reference the parameter, return false
        if(std::none_of(codeStrings.begin(), codeStrings.end(),
                        [&paramName](const std::string &c) 
                        { 
                            return (c.find("$(" + paramName + ")") != std::string::npos); 
                        }))
        {
            return false;
        }
        // Otherwise check if values are heterogeneous
        else {
            return isParamValueHeterogeneous<P>(index, getParamValuesFn);
        }
    }

    //------------------------------------------------------------------------
    // Protected members
    //------------------------------------------------------------------------
    // **YUCK**
    MergedStructGenerator<G, M> m_Gen;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const size_t m_Index;
    std::vector<std::reference_wrapper<const GroupInternal>> m_Groups;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronSpikeQueueUpdateGroupMerged : public GroupMerged<NeuronGroupInternal, NeuronSpikeQueueUpdateGroupMerged>
{
public:
    NeuronSpikeQueueUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecison, const BackendBase &backend,
                                      const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

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
class GENN_EXPORT NeuronGroupMergedBase : public GroupMerged<NeuronGroupInternal, NeuronGroupMergedBase>
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Should the parameter be implemented heterogeneously?
    bool isParamHeterogeneous(size_t index) const;

    //! Should the derived parameter be implemented heterogeneously?
    bool isDerivedParamHeterogeneous(size_t index) const;

    //! Should the var init parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const;

    //! Should the var init derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const;

    //! Should the current source parameter be implemented heterogeneously?
    bool isCurrentSourceParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the current source derived parameter be implemented heterogeneously?
    bool isCurrentSourceDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the current source var init parameter be implemented heterogeneously?
    bool isCurrentSourceVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the current source var init derived parameter be implemented heterogeneously?
    bool isCurrentSourceVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the postsynaptic model parameter be implemented heterogeneously?
    bool isPSMParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the postsynaptic model derived parameter be implemented heterogeneously?
    bool isPSMDerivedParamHeterogeneous(size_t childIndex, size_t varIndex) const;

    //! Should the GLOBALG postsynaptic model variable be implemented heterogeneously?
    bool isPSMGlobalVarHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the postsynaptic model var init parameter be implemented heterogeneously?
    bool isPSMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the postsynaptic model var init derived parameter be implemented heterogeneously?
    bool isPSMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    NeuronGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                          bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

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
    bool isChildParamValueHeterogeneous(std::initializer_list<std::string> codeStrings,
                                        const std::string &paramName, size_t childIndex, size_t paramIndex,
                                        const std::vector<std::vector<T>> &sortedGroupChildren, G getParamValuesFn) const
    {
        // If none of the code strings reference the parameter
        if(std::any_of(codeStrings.begin(), codeStrings.end(),
                        [&paramName](const std::string &c)
                        {
                            return (c.find("$(" + paramName + ")") != std::string::npos);
                        }))
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
        }

        return false;
    }

    template<typename T = NeuronGroupMergedBase, typename H, typename V>
    void addHeterogeneousChildParams(const Snippet::Base::StringVec &paramNames, size_t childIndex,
                                     const std::string &prefix, 
                                     H isChildParamHeterogeneousFn, V getValueFn)
    {
        // Loop through parameters
        for(size_t p = 0; p < paramNames.size(); p++) {
            // If parameter is heterogeneous
            if((static_cast<const T *>(this)->*isChildParamHeterogeneousFn)(childIndex, p)) {
                m_Gen.addScalarField(paramNames[p] + prefix + std::to_string(childIndex),
                                   [childIndex, p, getValueFn](const NeuronGroupInternal &, size_t groupIndex)
                                   {
                                       return Utils::writePreciseString(getValueFn(groupIndex, childIndex, p));
                                   });
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename H, typename V>
    void addHeterogeneousChildDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, size_t childIndex,
                                            const std::string &prefix, H isChildDerivedParamHeterogeneousFn, V getValueFn)
    {
        // Loop through derived parameters
        for(size_t p = 0; p < derivedParams.size(); p++) {
            // If parameter is heterogeneous
            if((static_cast<const T *>(this)->*isChildDerivedParamHeterogeneousFn)(childIndex, p)) {
                m_Gen.addScalarField(derivedParams[p].name + prefix + std::to_string(childIndex),
                                    [childIndex, p, getValueFn](const NeuronGroupInternal &, size_t groupIndex)
                                    {
                                        return Utils::writePreciseString(getValueFn(groupIndex, childIndex, p));
                                    });
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename H, typename V>
    void addHeterogeneousChildVarInitParams(const Snippet::Base::StringVec &paramNames, size_t childIndex,
                                            size_t varIndex, const std::string &prefix,
                                            H isChildParamHeterogeneousFn, V getVarInitialiserFn)
    {
        // Loop through parameters
        for(size_t p = 0; p < paramNames.size(); p++) {
            // If parameter is heterogeneous
            if((static_cast<const T*>(this)->*isChildParamHeterogeneousFn)(childIndex, varIndex, p)) {
                m_Gen.addScalarField(paramNames[p] + prefix + std::to_string(childIndex),
                                    [childIndex, varIndex, p, getVarInitialiserFn](const NeuronGroupInternal &, size_t groupIndex)
                                    {
                                        const std::vector<Models::VarInit> &varInit = getVarInitialiserFn(groupIndex, childIndex);
                                        return Utils::writePreciseString(varInit.at(varIndex).getParams().at(p));
                                    });
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename H, typename V>
    void addHeterogeneousChildVarInitDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, size_t childIndex,
                                                   size_t varIndex, const std::string &prefix,
                                                   H isChildDerivedParamHeterogeneousFn, V getVarInitialiserFn)
    {
        // Loop through parameters
        for(size_t p = 0; p < derivedParams.size(); p++) {
            // If parameter is heterogeneous
            if((static_cast<const T *>(this)->*isChildDerivedParamHeterogeneousFn)(childIndex, varIndex, p)) {
                m_Gen.addScalarField(derivedParams[p].name + prefix + std::to_string(childIndex),
                                    [childIndex, varIndex, p, getVarInitialiserFn](const NeuronGroupInternal &, size_t groupIndex)
                                    {
                                        const std::vector<Models::VarInit> &varInit = getVarInitialiserFn(groupIndex, childIndex);
                                        return Utils::writePreciseString(varInit.at(varIndex).getDerivedParams().at(p));
                                    });
            }
        }
    }

    template<typename S>
    void addChildEGPs(const std::vector<Snippet::Base::EGP> &egps, size_t childIndex,
                      const std::string &arrayPrefix, const std::string &prefix,
                      S getEGPSuffixFn)
    {
        using FieldType = std::remove_reference<decltype(m_Gen)>::type::FieldType;
        for(const auto &e : egps) {
            const bool isPointer = Utils::isTypePointer(e.type);
            const std::string varPrefix = isPointer ? arrayPrefix : "";
            m_Gen.addField(e.type, e.name + prefix + std::to_string(childIndex),
                          [getEGPSuffixFn, childIndex, e, varPrefix](const NeuronGroupInternal&, size_t groupIndex)
                          {
                              return varPrefix + e.name + getEGPSuffixFn(groupIndex, childIndex);
                          },
                          Utils::isTypePointer(e.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
        }
    }
    


    void addMergedInSynPointerField(const std::string &type, const std::string &name, 
                                    size_t archetypeIndex, const std::string &prefix);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<std::pair<SynapseGroupInternal *, std::vector<SynapseGroupInternal *>>>> m_SortedMergedInSyns;
    std::vector<std::vector<CurrentSourceInternal*>> m_SortedCurrentSources;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronUpdateGroupMerged : public NeuronGroupMergedBase
{
public:
    NeuronUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get the expression to calculate the queue offset for accessing state of variables this timestep
    std::string getCurrentQueueOffset() const;

    //! Get the expression to calculate the queue offset for accessing state of variables in previous timestep
    std::string getPrevQueueOffset() const;

    //! Should the incoming synapse weight update model parameter be implemented heterogeneously?
    bool isInSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the incoming synapse weight update model derived parameter be implemented heterogeneously?
    bool isInSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model parameter be implemented heterogeneously?
    bool isOutSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model derived parameter be implemented heterogeneously?
    bool isOutSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    //! Helper to generate merged struct fields for WU pre and post vars
    void generateWUVar(const BackendBase &backend, const std::string &fieldPrefixStem, 
                       const std::vector<SynapseGroupInternal*> &archetypeSyn,
                       const std::vector<std::vector<SynapseGroupInternal*>> &sortedSyn,
                       Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                       bool(NeuronUpdateGroupMerged::*isParamHeterogeneous)(size_t, size_t) const,
                       bool(NeuronUpdateGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t) const);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedInSynWithPostCode;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedOutSynWithPreCode;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronInitGroupMerged : public NeuronGroupMergedBase
{
public:
    NeuronInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                          const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //! Should the incoming synapse weight update model var init parameter be implemented heterogeneously?
    bool isInSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the incoming synapse weight update model var init derived parameter be implemented heterogeneously?
    bool isInSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model var init parameter be implemented heterogeneously?
    bool isOutSynWUMVarInitParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model var init derived parameter be implemented heterogeneously?
    bool isOutSynWUMVarInitDerivedParamHeterogeneous(size_t childIndex, size_t varIndex, size_t paramIndex) const;

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision,
                  const std::string &timePrecision) const;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    //! Helper to generate merged struct fields for WU pre and post vars
    void generateWUVar(const BackendBase &backend, const std::string &fieldPrefixStem,
                       const std::vector<SynapseGroupInternal *> &archetypeSyn,
                       const std::vector<std::vector<SynapseGroupInternal *>> &sortedSyn,
                       Models::Base::VarVec(WeightUpdateModels::Base::*getVars)(void) const,
                       const std::vector<Models::VarInit>&(SynapseGroupInternal::*getVarInitialisers)(void) const,
                       bool(NeuronInitGroupMerged::*isParamHeterogeneous)(size_t, size_t, size_t) const,
                       bool(NeuronInitGroupMerged::*isDerivedParamHeterogeneous)(size_t, size_t, size_t) const);


    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedInSynWithPostVars;
    std::vector<std::vector<SynapseGroupInternal *>> m_SortedOutSynWithPreVars;
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDendriticDelayUpdateGroupMerged : public GroupMerged<SynapseGroupInternal, SynapseDendriticDelayUpdateGroupMerged>
{
public:
    SynapseDendriticDelayUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                           const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &group);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision) const;
};

// ----------------------------------------------------------------------------
// SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityHostInitGroupMerged : public GroupMerged<SynapseGroupInternal, SynapseConnectivityHostInitGroupMerged>
{
public:
    SynapseConnectivityHostInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                           const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision) const;

    //! Should the connectivity initialization parameter be implemented heterogeneously for EGP init?
    bool isConnectivityInitParamHeterogeneous(size_t paramIndex) const;

    //! Should the connectivity initialization derived parameter be implemented heterogeneously for EGP init?
    bool isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const;
};

// ----------------------------------------------------------------------------
// SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityInitGroupMerged : public GroupMerged<SynapseGroupInternal, SynapseConnectivityInitGroupMerged>
{
public:
    SynapseConnectivityInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Should the connectivity initialization parameter be implemented heterogeneously?
    bool isConnectivityInitParamHeterogeneous(size_t paramIndex) const;

    //! Should the connectivity initialization parameter be implemented heterogeneously?
    bool isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const;

    void generate(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &precision) const;
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseGroupMergedBase : public GroupMerged<SynapseGroupInternal, SynapseGroupMergedBase>
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

    //! Should the weight update model parameter be implemented heterogeneously?
    bool isWUParamHeterogeneous(size_t paramIndex) const;

    //! Should the weight update model derived parameter be implemented heterogeneously?
    bool isWUDerivedParamHeterogeneous(size_t paramIndex) const;

    //! Should the GLOBALG weight update model variable be implemented heterogeneously?
    bool isWUGlobalVarHeterogeneous(size_t varIndex) const;

    //! Should the weight update model variable initialization parameter be implemented heterogeneously?
    bool isWUVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const;
    
    //! Should the weight update model variable initialization derived parameter be implemented heterogeneously?
    bool isWUVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const;

    //! Should the connectivity initialization parameter be implemented heterogeneously?
    bool isConnectivityInitParamHeterogeneous(size_t paramIndex) const;

    //! Should the connectivity initialization parameter be implemented heterogeneously?
    bool isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const;

    //! Is presynaptic neuron parameter heterogeneous
    bool isSrcNeuronParamHeterogeneous(size_t paramIndex) const;

    //! Is presynaptic neuron derived parameter heterogeneous
    bool isSrcNeuronDerivedParamHeterogeneous(size_t paramIndex) const;

    //! Is postsynaptic neuron parameter heterogeneous
    bool isTrgNeuronParamHeterogeneous(size_t paramIndex) const;

    //! Is postsynaptic neuron derived parameter heterogeneous
    bool isTrgNeuronDerivedParamHeterogeneous(size_t paramIndex) const;

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

    SynapseGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                           Role role, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual std::string getArchetypeCode() const { return ""; }

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
    void addPSPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addSrcPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addTrgPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addWeightSharingPointerField(const std::string &type, const std::string &name, const std::string &prefix);
};

//----------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PresynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    PresynapticUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::PresynapticUpdate, groups)
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

protected:
    //----------------------------------------------------------------------------
    // SynapseGroupMergedBase virtuals
    //----------------------------------------------------------------------------
    virtual std::string getArchetypeCode() const override
    {
        // **NOTE** we concatenate sim code, event code and threshold code so all get tested
        return getArchetype().getWUModel()->getSimCode() + getArchetype().getWUModel()->getEventCode() + getArchetype().getWUModel()->getEventThresholdConditionCode();
    }
};

//----------------------------------------------------------------------------
// CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PostsynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    PostsynapticUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                  const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::PostsynapticUpdate, groups)
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

protected:
    //----------------------------------------------------------------------------
    // SynapseGroupMergedBase virtuals
    //----------------------------------------------------------------------------
    virtual std::string getArchetypeCode() const override
    {
        return getArchetype().getWUModel()->getLearnPostCode();
    }
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDynamicsGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseDynamicsGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::SynapseDynamics, groups)
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

protected:
    //----------------------------------------------------------------------------
    // SynapseGroupMergedBase virtuals
    //----------------------------------------------------------------------------
    virtual std::string getArchetypeCode() const override
    {
        return getArchetype().getWUModel()->getSynapseDynamicsCode();
    }
};

//----------------------------------------------------------------------------
// CodeGenerator::SynapseDenseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDenseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseDenseInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::DenseInit, groups)
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
    SynapseSparseInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::SparseInit, groups)
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
