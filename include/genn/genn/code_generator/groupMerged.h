#pragma once

// Standard includes
#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "currentSourceInternal.h"
#include "customConnectivityUpdateInternal.h"
#include "customUpdateInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"
#include "type.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"

// GeNN transpiler includes
#include "transpiler/statement.h"
#include "transpiler/typeChecker.h"

// Forward declarations
namespace GeNN::CodeGenerator
{
class CodeStream;
}

namespace GeNN::Transpiler::TypeChecker
{
class EnvironmentBase;
}

//------------------------------------------------------------------------
// GeNN::CodeGenerator::GroupMergedFieldType
//------------------------------------------------------------------------
//! Enumeration of field types 
/*! The only reason this is not a child of GroupMerged is to prevent the 
    template nightmare that would otherwise ensue when declaring operators on it */
namespace GeNN::CodeGenerator
{
enum class GroupMergedFieldType : unsigned int
{
    STANDARD        = 0,
    HOST            = (1 << 0),
    DYNAMIC         = (1 << 1),

    HOST_DYNAMIC    = HOST | DYNAMIC,
};

//----------------------------------------------------------------------------
// Operators
//----------------------------------------------------------------------------
inline bool operator & (GroupMergedFieldType typeA, GroupMergedFieldType typeB)
{
    return (static_cast<unsigned int>(typeA) & static_cast<unsigned int>(typeB)) != 0;
}

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::ChildGroupMerged
//----------------------------------------------------------------------------
template<typename G>
class ChildGroupMerged
{
public:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef G GroupInternal;
    typedef std::function<std::string(const G &, size_t)> GetFieldValueFunc;
    typedef std::function<double(const G &, size_t)> GetFieldDoubleValueFunc;
    typedef std::tuple<Type::ResolvedType, std::string, GetFieldValueFunc, GroupMergedFieldType> Field;

    ChildGroupMerged(size_t index, const Type::TypeContext &typeContext, const std::vector<std::reference_wrapper<const GroupInternal>> groups)
    :   m_Index(index), m_TypeContext(typeContext), m_Groups(std::move(groups))
    {}

    ChildGroupMerged(const ChildGroupMerged&) = delete;
    ChildGroupMerged(ChildGroupMerged&&) = default;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get type context used to resolve any types involved in this group
    const Type::TypeContext &getTypeContext() const{ return m_TypeContext; }

    size_t getIndex() const { return m_Index; }

    //! Get 'archetype' neuron group - it's properties represent those of all other merged neuron groups
    const GroupInternal &getArchetype() const { return m_Groups.front().get(); }

    //! Gets access to underlying vector of neuron groups which have been merged
    const std::vector<std::reference_wrapper<const GroupInternal>> &getGroups() const{ return m_Groups; }

    const Type::ResolvedType &getScalarType() const{ return m_TypeContext.at("scalar"); }
    const Type::ResolvedType &getTimeType() const{ return m_TypeContext.at("timepoint"); }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    //! Helper to test whether parameter is referenced in vector of codestrings
    bool isParamReferenced(const std::vector<std::string> &codeStrings, const std::string &paramName) const
    {
        return std::any_of(codeStrings.begin(), codeStrings.end(),
                           [&paramName](const std::string &c)
                           {
                               return (c.find("$(" + paramName + ")") != std::string::npos);
                           });
    }

    //! Helper to test whether parameter values are heterogeneous within merged group
    template<typename P>
    bool isParamValueHeterogeneous(const std::string &name, P getParamValuesFn) const
    {
        // Get value of parameter in archetype group
        const double archetypeValue = getParamValuesFn(getArchetype()).at(name);

        // Return true if any parameter values differ from the archetype value
        return std::any_of(getGroups().cbegin(), getGroups().cend(),
                           [&name, archetypeValue, getParamValuesFn](const GroupInternal &g)
                           {
                               return (getParamValuesFn(g).at(name) != archetypeValue);
                           });
    }

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

    //! Helper to update hash with the hash of calling getHashableFn on each group
    template<typename H>
    void updateHash(H getHashableFn, boost::uuids::detail::sha1 &hash) const
    {
        for(const auto &g : getGroups()) {
            Utils::updateHash(getHashableFn(g.get()), hash);
        }
    }

    template<typename T, typename V, typename R>
    void updateParamHash(R isParamReferencedFn, V getValueFn, boost::uuids::detail::sha1 &hash) const
    {
        // Loop through parameters
        const auto &archetypeParams = getValueFn(getArchetype());
        for(const auto &p : archetypeParams) {
            // If any of the code strings reference the parameter
            if((static_cast<const T*>(this)->*isParamReferencedFn)(p.first)) {
                // Loop through groups
                for(const auto &g : getGroups()) {
                    // Update hash with parameter value
                    Utils::updateHash(getValueFn(g.get()).at(p.first), hash);
                }
            }
        }
    }

    template<typename T, typename A, typename R>
    void updateVarInitParamHash(R isParamReferencedFn, boost::uuids::detail::sha1 &hash) const
    {
        // Loop through variables
        const auto &archetypeVarInitialisers = A(getArchetype()).getInitialisers();
        for(const auto &varInit : archetypeVarInitialisers) {
            // Loop through parameters
            for(const auto &p : varInit.second.getParams()) {
                // If any of the code strings reference the parameter
                if((static_cast<const T *>(this)->*isParamReferencedFn)(varInit.first, p.first)) {
                    // Loop through groups
                    for(const auto &g : getGroups()) {
                        const auto &values = A(g.get()).getInitialisers().at(varInit.first).getParams();

                        // Update hash with parameter value
                        Utils::updateHash(values.at(p.first), hash);
                    }
                }
            }
        }
    }

    template<typename T, typename A, typename R>
    void updateVarInitDerivedParamHash(R isDerivedParamReferencedFn, boost::uuids::detail::sha1 &hash) const
    {
        // Loop through variables
        const auto &archetypeVarInitialisers = A(getArchetype()).getInitialisers();
        for(const auto &varInit : archetypeVarInitialisers) {
            // Loop through parameters
            for(const auto &d : varInit.second.getDerivedParams()) {
                // If any of the code strings reference the parameter
                if((static_cast<const T *>(this)->*isDerivedParamReferencedFn)(varInit.first, d.first)) {
                    // Loop through groups
                    for(const auto &g : getGroups()) {
                        const auto &values = A(g.get()).getInitialisers().at(varInit.first).getDerivedParams();

                        // Update hash with parameter value
                        Utils::updateHash(values.at(d.first), hash);
                    }
                }
            }
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    size_t m_Index;
    const Type::TypeContext &m_TypeContext;
    std::vector<std::reference_wrapper<const GroupInternal>> m_Groups;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::GroupMerged
//----------------------------------------------------------------------------
//! Very thin wrapper around a number of groups which have been merged together
template<typename G>
class GroupMerged : public ChildGroupMerged<G>
{
public:
    GroupMerged(size_t index, const Type::TypeContext &typeContext, const std::vector<std::reference_wrapper<const GroupInternal>> groups)
    :   ChildGroupMerged<G>(index, typeContext, std::move(groups))
    {}

    GroupMerged(const GroupMerged&) = delete;
    GroupMerged(GroupMerged&&) = default;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Get name of memory space assigned to group
    const std::string &getMemorySpace() const { return m_MemorySpace; }

    //! Get group fields
    const std::vector<Field> &getFields() const{ return m_Fields; }

    //! Get group fields, sorted into order they will appear in struct
    std::vector<Field> getSortedFields(const BackendBase &backend) const
    {
        // Make a copy of fields and sort so largest come first. This should mean that due
        // to structure packing rules, significant memory is saved and estimate is more precise
        auto sortedFields = m_Fields;
        const size_t pointerBytes = backend.getPointerBytes();
        std::sort(sortedFields.begin(), sortedFields.end(),
                  [pointerBytes](const Field &a, const Field &b)
                  {
                      return (std::get<0>(a).getSize(pointerBytes) > std::get<0>(b).getSize(pointerBytes));
                  });
        return sortedFields;

    }

    //! Generate declaration of struct to hold this merged group
    void generateStruct(CodeStream &os, const BackendBase &backend, const std::string &name, bool host = false) const
    {
        os << "struct Merged" << name << "Group" << getIndex() << std::endl;
        {
            // Loop through fields and write to structure
            CodeStream::Scope b(os);
            const auto sortedFields = getSortedFields(backend);
            for(const auto &f : sortedFields) {
                // If field is a pointer and not marked as being a host field 
                // (in which case the backend should leave its type alone!)
                const auto &type = std::get<0>(f);
                if(type.isPointer() && !(std::get<3>(f) & GroupMergedFieldType::HOST)) {
                    // If we are generating a host structure, allow the backend to override the type
                    if(host) {
                        os << backend.getMergedGroupFieldHostTypeName(type);
                    }
                    // Otherwise, allow the backend to add a prefix 
                    else {
                        os << backend.getPointerPrefix() << type.getName();
                    }
                }
                // Otherwise, leave the type alone
                else {
                    os << type.getName();
                }
                os << " " << std::get<1>(f) << ";" << std::endl;
            }
            os << std::endl;
        }

        os << ";" << std::endl;
    }

    void generateStructFieldArgumentDefinitions(CodeStream &os, const BackendBase &backend) const
    {
        // Get sorted fields
        const auto sortedFields = getSortedFields(backend);
        for(size_t fieldIndex = 0; fieldIndex < sortedFields.size(); fieldIndex++) {
            const auto &f = sortedFields[fieldIndex];
            os << backend.getMergedGroupFieldHostTypeName(std::get<0>(f)) << " " << std::get<1>(f);
            if(fieldIndex != (sortedFields.size() - 1)) {
                os << ", ";
            }
        }
    }

    size_t getStructArraySize(const BackendBase &backend) const
    {
        // Loop through fields again to generate any EGP pushing functions that are required and to calculate struct size
        size_t structSize = 0;
        size_t largestFieldSize = 0;
        const auto sortedFields = getSortedFields(backend);
        for(const auto &f : sortedFields) {
            // Add size of field to total
            const size_t fieldSize = std::get<0>(f).getSize(backend.getPointerBytes());
            structSize += fieldSize;

            // Update largest field size
            largestFieldSize = std::max(fieldSize, largestFieldSize);
        }

        // Add total size of array of merged structures to merged struct data
        // **NOTE** to match standard struct packing rules we pad to a multiple of the largest field size
        return padSize(structSize, largestFieldSize) * getGroups().size();
    }

    //! Assign memory spaces to group
    /*! Memory spaces are given out on a first-come, first-serve basis so this should be called on groups in preferential order */
    void assignMemorySpaces(const BackendBase &backend, BackendBase::MemorySpaces &memorySpaces)
    {
        // If this backend uses memory spaces
        if(!memorySpaces.empty()) {
            // Get size of group in bytes
            const size_t groupBytes = getStructArraySize(backend);

            // Loop through memory spaces
            for(auto &m : memorySpaces) {
                // If there is space in this memory space for group
                if(m.second > groupBytes) {
                    // Cache memory space name in object
                    m_MemorySpace = m.first;

                    // Subtract
                    m.second -= groupBytes;

                    // Stop searching
                    break;
                }
            }
        }
    }

    void addField(const Type::ResolvedType &type, const std::string &name, GetFieldValueFunc getFieldValue, GroupMergedFieldType fieldType = GroupMergedFieldType::STANDARD)
    {
        // Add field to data structure
        m_Fields.emplace_back(type, name, getFieldValue, fieldType);
    }

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    void generateRunnerBase(const BackendBase &backend, 
                            CodeStream &definitionsInternal, CodeStream &definitionsInternalFunc, 
                            CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, 
                            CodeStream &runnerMergedStructAlloc, const std::string &name, bool host = false) const
    {
        // Make a copy of fields and sort so largest come first. This should mean that due
        // to structure packing rules, significant memory is saved and estimate is more precise
        auto sortedFields = getSortedFields(backend);

        // If this isn't a host merged structure, generate definition for function to push group
        if(!host) {
            definitionsInternalFunc << "EXPORT_FUNC void pushMerged" << name << "Group" << getIndex() << "ToDevice(unsigned int idx, ";
            generateStructFieldArgumentDefinitions(definitionsInternalFunc, backend);
            definitionsInternalFunc << ");" << std::endl;
        }

        // Loop through fields again to generate any EGP pushing functions that are require
        for(const auto &f : sortedFields) {
            // If this field is a dynamic pointer
            if((std::get<3>(f) & GroupMergedFieldType::DYNAMIC) && std::get<0>(f).isPointer()) {
                definitionsInternalFunc << "EXPORT_FUNC void pushMerged" << name << getIndex() << std::get<1>(f) << "ToDevice(unsigned int idx, ";
                definitionsInternalFunc << backend.getMergedGroupFieldHostTypeName(std::get<0>(f)) << " value);" << std::endl;
            }

            // Raise error if this field is a host field but this isn't a host structure
            assert(!(std::get<3>(f) & GroupMergedFieldType::HOST) || host);
        }

        // If merged group is used on host
        if(host) {
            // Generate struct directly into internal definitions
            // **NOTE** we ignore any backend prefix as we're generating this struct for use on the host
            generateStruct(definitionsInternal, backend, name, true);

            // Declare array of these structs containing individual neuron group pointers etc
            runnerVarDecl << "Merged" << name << "Group" << getIndex() << " merged" << name << "Group" << getIndex() << "[" << getGroups().size() << "];" << std::endl;

            // Export it
            definitionsInternalVar << "EXPORT_VAR Merged" << name << "Group" << getIndex() << " merged" << name << "Group" << getIndex() << "[" << getGroups().size() << "]; " << std::endl;
        }

        // Loop through groups
        for(size_t groupIndex = 0; groupIndex < getGroups().size(); groupIndex++) {
            // If this is a merged group used on the host, directly set array entry
            if(host) {
                runnerMergedStructAlloc << "merged" << name << "Group" << getIndex() << "[" << groupIndex << "] = {";
                generateStructFieldArguments(runnerMergedStructAlloc, groupIndex, sortedFields);
                runnerMergedStructAlloc << "};" << std::endl;
            }
            // Otherwise, call function to push to device
            else {
                runnerMergedStructAlloc << "pushMerged" << name << "Group" << getIndex() << "ToDevice(" << groupIndex << ", ";
                generateStructFieldArguments(runnerMergedStructAlloc, groupIndex, sortedFields);
                runnerMergedStructAlloc << ");" << std::endl;
            }
        }
    }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void generateStructFieldArguments(CodeStream &os, size_t groupIndex, 
                                      const std::vector<Field> &sortedFields) const
    {
        // Get group by index
        const auto &g = getGroups()[groupIndex];

        // Loop through fields
        for(size_t fieldIndex = 0; fieldIndex < sortedFields.size(); fieldIndex++) {
            const auto &f = sortedFields[fieldIndex];
            const std::string fieldInitVal = std::get<2>(f)(g, groupIndex);
            os << fieldInitVal;
            if(fieldIndex != (sortedFields.size() - 1)) {
                os << ", ";
            }
        }
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_MemorySpace;
    std::vector<Field> m_Fields;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronSpikeQueueUpdateGroupMerged : public GroupMerged<NeuronGroupInternal>
{
public:
    NeuronSpikeQueueUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                      const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

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

    void genMergedGroupSpikeCountReset(EnvironmentExternalBase &env, unsigned int batchSize) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronPrevSpikeTimeUpdateGroupMerged : public GroupMerged<NeuronGroupInternal>
{
public:
    NeuronPrevSpikeTimeUpdateGroupMerged(size_t index, const Type::TypeContext &typeContext, const BackendBase &backend,
                                         const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

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

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::NeuronGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronGroupMergedBase : public GroupMerged<NeuronGroupInternal>
{
public:
    using GroupMerged::GroupMerged;

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    template<typename M, typename G, typename H>
    void orderNeuronGroupChildren(std::vector<M> &childGroups, const Type::TypeContext &typeContext, G getVectorFunc, H getHashDigestFunc) const
    {
        const std::vector<typename M::GroupInternal*> &archetypeChildren = (getArchetype().*getVectorFunc)();

        // Resize vector of vectors to hold children for all neuron groups, sorted in a consistent manner
        std::vector<std::vector<std::reference_wrapper<typename M::GroupInternal const>>> sortedGroupChildren;
        sortedGroupChildren.resize(archetypeChildren.size());

        // Create temporary vector of children and their digests
        std::vector<std::pair<boost::uuids::detail::sha1::digest_type, M::GroupInternal*>> childDigests;
        childDigests.reserve(archetypeChildren.size());

        // Loop through groups
        for(const auto &g : getGroups()) {
            // Get group children
            const std::vector<typename M::GroupInternal*> &groupChildren = (g.get().*getVectorFunc)();
            assert(groupChildren.size() == archetypeChildren.size());

            // Loop through children and add them and their digests to vector
            childDigests.clear();
            for(auto *c : groupChildren) {
                childDigests.emplace_back((c->*getHashDigestFunc)(), c);
            }

            // Sort by digest
            std::sort(childDigests.begin(), childDigests.end(),
                      [](const auto &a, const auto &b)
                      {
                          return (a.first < b.first);
                      });


            // Populate 'transpose' vector of vectors
            for (size_t i = 0; i < childDigests.size(); i++) {
                sortedGroupChildren[i].emplace_back(*childDigests[i].second);
            }
        }

        // Reserve vector of child groups and create merged group objects based on vector of groups
        childGroups.reserve(archetypeChildren.size());
        for(size_t i = 0; i < sortedGroupChildren.size(); i++) {
            childGroups.emplace_back(i, typeContext, sortedGroupChildren[i]);
        }
    }

    //! Is the var init parameter referenced?
    bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;
};

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::SynapseGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseGroupMergedBase : public GroupMerged<SynapseGroupInternal>
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Should the weight update model parameter be implemented heterogeneously?
    bool isWUParamHeterogeneous(const std::string &paramName) const;

    //! Should the weight update model derived parameter be implemented heterogeneously?
    bool isWUDerivedParamHeterogeneous(const std::string &paramName) const;

    //! Should the GLOBALG weight update model variable be implemented heterogeneously?
    bool isWUGlobalVarHeterogeneous(const std::string &varName) const;

    //! Should the weight update model variable initialization parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;
    
    //! Should the weight update model variable initialization derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

    //! Should the sparse connectivity initialization parameter be implemented heterogeneously?
    bool isSparseConnectivityInitParamHeterogeneous(const std::string &paramName) const;

    //! Should the sparse connectivity initialization parameter be implemented heterogeneously?
    bool isSparseConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const;

    //! Should the Toeplitz connectivity initialization parameter be implemented heterogeneously?
    bool isToeplitzConnectivityInitParamHeterogeneous(const std::string &paramName) const;

    //! Should the Toeplitz connectivity initialization parameter be implemented heterogeneously?
    bool isToeplitzConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const;

    //! Is presynaptic neuron parameter heterogeneous?
    bool isSrcNeuronParamHeterogeneous(const std::string &paramName) const;

    //! Is presynaptic neuron derived parameter heterogeneous?
    bool isSrcNeuronDerivedParamHeterogeneous(const std::string &paramName) const;

    //! Is postsynaptic neuron parameter heterogeneous?
    bool isTrgNeuronParamHeterogeneous(const std::string &paramName) const;

    //! Is postsynaptic neuron derived parameter heterogeneous?
    bool isTrgNeuronDerivedParamHeterogeneous(const std::string &paramName) const;

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

    std::string getPreSlot(unsigned int batchSize) const;
    std::string getPostSlot(unsigned int batchSize) const;

    std::string getPreVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
    {
        return getPreVarIndex(getArchetype().getSrcNeuronGroup()->isDelayRequired(), batchSize, varDuplication, index);
    }
    
    std::string getPostVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
    {
        return getPostVarIndex(getArchetype().getTrgNeuronGroup()->isDelayRequired(), batchSize, varDuplication, index);
    }

    std::string getPreWUVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
    {
        return getPreVarIndex(getArchetype().getDelaySteps() != 0, batchSize, varDuplication, index);
    }
    
    std::string getPostWUVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const
    {
        return getPostVarIndex(getArchetype().getBackPropDelaySteps() != 0, batchSize, varDuplication, index);
    }

    std::string getPostDenDelayIndex(unsigned int batchSize, const std::string &index, const std::string &offset) const;

    std::string getPreVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getPostVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;

    std::string getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    
    std::string getPostISynIndex(unsigned int batchSize, const std::string &index) const
    {
        return ((batchSize == 1) ? "" : "postBatchOffset + ") + index;
    }

    std::string getPreISynIndex(unsigned int batchSize, const std::string &index) const
    {
        return ((batchSize == 1) ? "" : "preBatchOffset + ") + index;
    }

    std::string getSynVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getKernelVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    
protected:
    //----------------------------------------------------------------------------
    // Enumerations
    //----------------------------------------------------------------------------
    enum class Role
    {
        PresynapticUpdate,
        PostsynapticUpdate,
        SynapseDynamics,
        Init,
        SparseInit,
        ConnectivityInit,
    };

    SynapseGroupMergedBase(size_t index, const Type::TypeContext &typeContext,
                           Role role, const std::string &archetypeCode, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   GroupMerged<SynapseGroupInternal>(index, typeContext, groups), m_ArchetypeCode(archetypeCode)
    {}

    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest(Role role) const;


private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    std::string getVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication,
                            const std::string &index, const std::string &prefix) const;
    
    //! Is the weight update model parameter referenced?
    bool isWUParamReferenced(const std::string &paramName) const;

    //! Is the GLOBALG weight update model variable referenced?
    bool isWUGlobalVarReferenced(const std::string &varName) const;

    //! Is the weight update model variable initialization parameter referenced?
    bool isWUVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;

    //! Is the sparse connectivity initialization parameter referenced?
    bool isSparseConnectivityInitParamReferenced(const std::string &paramName) const;

    //! Is the toeplitz connectivity initialization parameter referenced?
    bool isToeplitzConnectivityInitParamReferenced(const std::string &paramName) const;

    //! Is presynaptic neuron parameter referenced?
    bool isSrcNeuronParamReferenced(const std::string &paramName) const;

    //! Is postsynaptic neuron parameter referenced?
    bool isTrgNeuronParamReferenced(const std::string &paramName) const;

    static const std::vector<unsigned int>& getGroupKernelSize(const SynapseGroupInternal& g)
    {
        return g.getKernelSize();
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_ArchetypeCode;
};
}   // namespace GeNN::CodeGenerator
