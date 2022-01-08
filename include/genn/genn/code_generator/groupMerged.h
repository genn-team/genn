#pragma once

// Standard includes
#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

// GeNN includes
#include "gennExport.h"
#include "currentSourceInternal.h"
#include "customUpdateInternal.h"
#include "neuronGroupInternal.h"
#include "synapseGroupInternal.h"

// GeNN code generator includes
#include "code_generator/backendBase.h"
#include "code_generator/codeGenUtils.h"

// Forward declarations
namespace CodeGenerator
{
class CodeStream;
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
    // Enumerations
    //------------------------------------------------------------------------
    enum class FieldType
    {
        Standard,
        Host,
        ScalarEGP,
        PointerEGP,
    };

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef G GroupInternal;
    typedef std::function<std::string(const G &, size_t)> GetFieldValueFunc;
    typedef std::tuple<std::string, std::string, GetFieldValueFunc, FieldType> Field;


    GroupMerged(size_t index, const std::string &precision, const std::vector<std::reference_wrapper<const GroupInternal>> groups)
    :   m_Index(index), m_LiteralSuffix((precision == "float") ? "f" : ""), m_Groups(std::move(groups))
    {}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getIndex() const { return m_Index; }

    //! Get 'archetype' neuron group - it's properties represent those of all other merged neuron groups
    const GroupInternal &getArchetype() const { return m_Groups.front().get(); }

    //! Get name of memory space assigned to group
    const std::string &getMemorySpace() const { return m_MemorySpace; }

    //! Gets access to underlying vector of neuron groups which have been merged
    const std::vector<std::reference_wrapper<const GroupInternal>> &getGroups() const{ return m_Groups; }

    //! Get group fields
    const std::vector<Field> &getFields() const{ return m_Fields; }

    //! Get group fields, sorted into order they will appear in struct
    std::vector<Field> getSortedFields(const BackendBase &backend) const
    {
        // Make a copy of fields and sort so largest come first. This should mean that due
        // to structure packing rules, significant memory is saved and estimate is more precise
        auto sortedFields = m_Fields;
        std::sort(sortedFields.begin(), sortedFields.end(),
                  [&backend](const Field &a, const Field &b)
                  {
                      return (backend.getSize(std::get<0>(a)) > backend.getSize(std::get<0>(b)));
                  });
        return sortedFields;

    }

    //! Generate declaration of struct to hold this merged group
    void generateStruct(CodeStream &os, const BackendBase &backend, const std::string &name,
                        bool host = false) const
    {
        os << "struct Merged" << name << "Group" << getIndex() << std::endl;
        {
            // Loop through fields and write to structure
            CodeStream::Scope b(os);
            const auto sortedFields = getSortedFields(backend);
            for(const auto &f : sortedFields) {
                // If field is a pointer and not marked as being a host field 
                // (in which case the backend should leave its type alone!)
                const std::string &type = std::get<0>(f);
                if(::Utils::isTypePointer(type) && std::get<3>(f) != FieldType::Host) {
                    // If we are generating a host structure, allow the backend to override the type
                    if(host) {
                        os << backend.getMergedGroupFieldHostType(type);
                    }
                    // Otherwise, allow the backend to add a prefix 
                    else {
                        os << backend.getPointerPrefix() << type;
                    }
                }
                // Otherwise, leave the type alone
                else {
                    os << type;
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
            os << backend.getMergedGroupFieldHostType(std::get<0>(f)) << " " << std::get<1>(f);
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
            const size_t fieldSize = backend.getSize(std::get<0>(f));
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

protected:
    //------------------------------------------------------------------------
    // Protected methods
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

    void addField(const std::string &type, const std::string &name, GetFieldValueFunc getFieldValue, FieldType fieldType = FieldType::Standard)
    {
        // Add field to data structure
        m_Fields.emplace_back(type, name, getFieldValue, fieldType);
    }

    void addScalarField(const std::string &name, GetFieldValueFunc getFieldValue, FieldType fieldType = FieldType::Standard)
    {
        addField("scalar", name,
                 [getFieldValue, this](const G &g, size_t i)
                 {
                     return getFieldValue(g, i) + m_LiteralSuffix;
                 },
                 fieldType);
    }

    void addPointerField(const std::string &type, const std::string &name, const std::string &prefix)
    {
        assert(!Utils::isTypePointer(type));
        addField(type + "*", name, [prefix](const G &g, size_t) { return prefix + g.getName(); });
    }


    void addVars(const Models::Base::VarVec &vars, const std::string &arrayPrefix)
    {
        // Loop through variables
        for(const auto &v : vars) {
            addPointerField(v.type, v.name, arrayPrefix + v.name);
        }
    }

    template<typename V>
    void addVarReferences(const Models::Base::VarRefVec &varReferences, const std::string &arrayPrefix, V getVarRefFn)
    {
        // Loop through variables
        for(const auto &v : varReferences) {
            addField(v.type + "*", v.name, 
                     [getVarRefFn, arrayPrefix, v](const G &g, size_t) 
                     { 
                         const auto varRef = getVarRefFn(g).at(v.name);
                         return arrayPrefix + varRef.getVar().name + varRef.getTargetName(); 
                     });
        }
    }

    void addEGPs(const Snippet::Base::EGPVec &egps, const std::string &arrayPrefix, const std::string &varName = "")
    {
        for(const auto &e : egps) {
            const bool isPointer = Utils::isTypePointer(e.type);
            const std::string prefix = isPointer ? arrayPrefix : "";
            addField(e.type, e.name + varName,
                     [e, prefix, varName](const G &g, size_t) { return prefix + e.name + varName + g.getName(); },
                     isPointer ? FieldType::PointerEGP : FieldType::ScalarEGP);
        }
    }

    template<typename T, typename P, typename H>
    void addHeterogeneousParams(const Snippet::Base::StringVec &paramNames, const std::string &suffix,
                                P getParamValues, H isHeterogeneous)
    {
        // Loop through params
        for(const auto &p : paramNames) {
            // If parameters is heterogeneous
            if((static_cast<const T*>(this)->*isHeterogeneous)(p)) {
                // Add field
                addScalarField(p + suffix,
                               [p, getParamValues](const G &g, size_t)
                               {
                                   const auto &values = getParamValues(g);
                                   return Utils::writePreciseString(values.at(p));
                               });
            }
        }
    }

    template<typename T, typename D, typename H>
    void addHeterogeneousDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, const std::string &suffix,
                                       D getDerivedParamValues, H isHeterogeneous)
    {
        // Loop through derived params
        for(const auto &d : derivedParams) {
            // If parameters isn't homogeneous
            if((static_cast<const T*>(this)->*isHeterogeneous)(d.name)) {
                // Add field
                addScalarField(d.name + suffix,
                               [d, getDerivedParamValues](const G &g, size_t)
                               {
                                   const auto &values = getDerivedParamValues(g);
                                   return Utils::writePreciseString(values.at(d.name));
                               });
            }
        }
    }

    template<typename T, typename V, typename H>
    void addHeterogeneousVarInitParams(V getVarInitialisers, H isHeterogeneous)
    {
        // Loop through weight update model variables
        const auto &archetypeVarInitialisers = (getArchetype().*getVarInitialisers)();
        for(const auto &varInit : archetypeVarInitialisers) {
            // Loop through parameters
            for(const auto &p : varInit.second.getSnippet()->getParamNames()) {
                if((static_cast<const T*>(this)->*isHeterogeneous)(varInit.first, p)) {
                    addScalarField(p + varInit.first,
                                   [p, varInit, getVarInitialisers](const G &g, size_t)
                                   {
                                       const auto &values = (g.*getVarInitialisers)().at(varInit.first).getParams();
                                       return Utils::writePreciseString(values.at(p));
                                   });
                }
            }
        }
    }

    template<typename T, typename V, typename H>
    void addHeterogeneousVarInitDerivedParams(V getVarInitialisers, H isHeterogeneous)
    {
        // Loop through weight update model variables
        const auto &archetypeVarInitialisers = (getArchetype().*getVarInitialisers)();
        for(const auto &varInit : archetypeVarInitialisers) {
            // Loop through parameters
            for(const auto &d : varInit.second.getSnippet()->getDerivedParams()) {
                if((static_cast<const T*>(this)->*isHeterogeneous)(varInit.first, d.name)) {
                    addScalarField(d.name + varInit.first,
                                   [d, varInit, getVarInitialisers](const G &g, size_t)
                                   {
                                       const auto &values = (g.*getVarInitialisers)().at(varInit.first).getDerivedParams();
                                       return Utils::writePreciseString(values.at(d.name));
                                   });
                }
            }
        }
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

    template<typename T, typename V, typename R>
    void updateVarInitParamHash(V getVarInitialisers, R isParamReferencedFn, boost::uuids::detail::sha1 &hash) const
    {
        // Loop through variables
        const auto &archetypeVarInitialisers = (getArchetype().*getVarInitialisers)();
        for(const auto &varInit : archetypeVarInitialisers) {
            // Loop through parameters
            for(const auto &p : varInit.second.getParams()) {
                // If any of the code strings reference the parameter
                if((static_cast<const T *>(this)->*isParamReferencedFn)(varInit.first, p.first)) {
                    // Loop through groups
                    for(const auto &g : getGroups()) {
                        const auto &values = (g.get().*getVarInitialisers)().at(varInit.first).getParams();

                        // Update hash with parameter value
                        Utils::updateHash(values.at(p.first), hash);
                    }
                }
            }
        }
    }

    template<typename T, typename V, typename R>
    void updateVarInitDerivedParamHash(V getVarInitialisers, R isDerivedParamReferencedFn, boost::uuids::detail::sha1 &hash) const
    {
        // Loop through variables
        const auto &archetypeVarInitialisers = (getArchetype().*getVarInitialisers)();
        for(const auto &varInit : archetypeVarInitialisers) {
            // Loop through parameters
            for(const auto &d : varInit.second.getDerivedParams()) {
                // If any of the code strings reference the parameter
                if((static_cast<const T *>(this)->*isDerivedParamReferencedFn)(varInit.first, d.first)) {
                    // Loop through groups
                    for(const auto &g : getGroups()) {
                        const auto &values = (g.get().*getVarInitialisers)().at(varInit.first).getDerivedParams();

                        // Update hash with parameter value
                        Utils::updateHash(values.at(d.first), hash);
                    }
                }
            }
        }
    }

    void generateRunnerBase(const BackendBase &backend, CodeStream &definitionsInternal,
                            CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                            CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                            const std::string &name, bool host = false) const
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
            // If this field is for a pointer EGP, also declare function to push it
            if(std::get<3>(f) == FieldType::PointerEGP) {
                definitionsInternalFunc << "EXPORT_FUNC void pushMerged" << name << getIndex() << std::get<1>(f) << "ToDevice(unsigned int idx, ";
                definitionsInternalFunc << backend.getMergedGroupFieldHostType(std::get<0>(f)) << " value);" << std::endl;
            }

            // Raise error if this field is a host field but this isn't a host structure
            assert(std::get<3>(f) != FieldType::Host || host);
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
    const size_t m_Index;
    const std::string m_LiteralSuffix;
    std::string m_MemorySpace;
    std::vector<Field> m_Fields;
    std::vector<std::reference_wrapper<const GroupInternal>> m_Groups;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronSpikeQueueUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronSpikeQueueUpdateGroupMerged : public GroupMerged<NeuronGroupInternal>
{
public:
    NeuronSpikeQueueUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecison, const BackendBase &backend,
                                      const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    void genMergedGroupSpikeCountReset(CodeStream &os, unsigned int batchSize) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

//----------------------------------------------------------------------------
// CodeGenerator::NeuronPrevSpikeTimeUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronPrevSpikeTimeUpdateGroupMerged : public GroupMerged<NeuronGroupInternal>
{
public:
    NeuronPrevSpikeTimeUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecison, const BackendBase &backend,
                                         const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
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
// CodeGenerator::NeuronGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT NeuronGroupMergedBase : public GroupMerged<NeuronGroupInternal>
{
public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    //! Should the parameter be implemented heterogeneously?
    bool isParamHeterogeneous(const std::string &paramName) const;

    //! Should the derived parameter be implemented heterogeneously?
    bool isDerivedParamHeterogeneous(const std::string &paramName) const;

    //! Should the var init parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

    //! Should the var init derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

    //! Should the current source parameter be implemented heterogeneously?
    bool isCurrentSourceParamHeterogeneous(size_t childIndex, const std::string &paramName) const;

    //! Should the current source derived parameter be implemented heterogeneously?
    bool isCurrentSourceDerivedParamHeterogeneous(size_t childIndex, const std::string &paramName) const;

    //! Should the current source var init parameter be implemented heterogeneously?
    bool isCurrentSourceVarInitParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Should the current source var init derived parameter be implemented heterogeneously?
    bool isCurrentSourceVarInitDerivedParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Should the postsynaptic model parameter be implemented heterogeneously?
    bool isPSMParamHeterogeneous(size_t childIndex, const std::string &paramName) const;

    //! Should the postsynaptic model derived parameter be implemented heterogeneously?
    bool isPSMDerivedParamHeterogeneous(size_t childIndex, const std::string &paramName) const;

    //! Should the GLOBALG postsynaptic model variable be implemented heterogeneously?
    bool isPSMGlobalVarHeterogeneous(size_t childIndex, const std::string &varName) const;

    //! Should the postsynaptic model var init parameter be implemented heterogeneously?
    bool isPSMVarInitParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Should the postsynaptic model var init derived parameter be implemented heterogeneously?
    bool isPSMVarInitDerivedParamHeterogeneous(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Get sorted vectors of merged incoming synapse groups belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeMergedInSyns() const { return m_SortedMergedInSyns.front(); }

    //! Get sorted vectors of merged outgoing synapse groups with presynaptic output belonging to archetype group
    const std::vector<SynapseGroupInternal*> &getSortedArchetypeMergedPreOutputOutSyns() const { return m_SortedMergedPreOutputOutSyns.front(); }

    //! Get sorted vectors of current sources belonging to archetype group
    const std::vector<CurrentSourceInternal*> &getSortedArchetypeCurrentSources() const { return m_SortedCurrentSources.front(); }

protected:
    //------------------------------------------------------------------------
    // Protected methods
    //------------------------------------------------------------------------
    NeuronGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                          bool init, const std::vector<std::reference_wrapper<const NeuronGroupInternal>> &groups);

    void updateBaseHash(bool init, boost::uuids::detail::sha1 &hash) const;

    template<typename T, typename G, typename H>
    void orderNeuronGroupChildren(std::vector<std::vector<T*>> &sortedGroupChildren,
                                  G getVectorFunc, H getHashDigestFunc) const
    {
        const std::vector<T*> &archetypeChildren = (getArchetype().*getVectorFunc)();

        // Reserve vector of vectors to hold children for all neuron groups, in archetype order
        sortedGroupChildren.reserve(getGroups().size());

        // Create temporary vector of children and their digests
        std::vector<std::pair<boost::uuids::detail::sha1::digest_type, T*>> childDigests;
        childDigests.reserve(archetypeChildren.size());

        // Loop through groups
        for(const auto &g : getGroups()) {
            // Get group children
            const std::vector<T*> &groupChildren = (g.get().*getVectorFunc)();
            assert(groupChildren.size() == archetypeChildren.size());

            // Loop through children and add them and their digests to vector
            childDigests.clear();
            for(auto *c : groupChildren) {
                childDigests.emplace_back((c->*getHashDigestFunc)(), c);
            }

            // Sort by digest
            std::sort(childDigests.begin(), childDigests.end(),
                      [](const std::pair<boost::uuids::detail::sha1::digest_type, T*> &a,
                         const std::pair<boost::uuids::detail::sha1::digest_type, T*> &b)
                      {
                          return (a.first < b.first);
                      });


            // Reserve vector for this group's children
            sortedGroupChildren.emplace_back();
            sortedGroupChildren.back().reserve(groupChildren.size());

            // Copy sorted child pointers into sortedGroupChildren
            std::transform(childDigests.cbegin(), childDigests.cend(), std::back_inserter(sortedGroupChildren.back()),
                           [](const std::pair<boost::uuids::detail::sha1::digest_type, T*> &a){ return a.second; });
        }
    }

    //! Is the var init parameter referenced?
    bool isVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;

    //! Is the var init derived parameter referenced?
    bool isVarInitDerivedParamReferenced(const std::string &varName, const std::string &paramName) const;

    //! Is the current source parameter referenced?
    bool isCurrentSourceParamReferenced(size_t childIndex, const std::string &paramName) const;

    //! Is the current source derived parameter referenced?
    bool isCurrentSourceDerivedParamReferenced(size_t childIndex, const std::string &paramName) const;

    //! Is the current source var init parameter referenced?
    bool isCurrentSourceVarInitParamReferenced(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Is the current source var init derived parameter referenced?
    bool isCurrentSourceVarInitDerivedParamReferenced(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Is the postsynaptic model parameter referenced?
    bool isPSMParamReferenced(size_t childIndex, const std::string &paramName) const;

    //! Is the postsynaptic model derived parameter referenced?
    bool isPSMDerivedParamReferenced(size_t childIndex, const std::string &paramName) const;

    //! Is the GLOBALG postsynaptic model variable referenced?
    bool isPSMGlobalVarReferenced(size_t childIndex, const std::string &varName) const;

    //! Is the postsynaptic model var init parameter referenced?
    bool isPSMVarInitParamReferenced(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    //! Is the postsynaptic model var init derived parameter referenced?
    bool isPSMVarInitDerivedParamReferenced(size_t childIndex, const std::string &varName, const std::string &paramName) const;

    template<typename T, typename G>
    bool isChildParamValueHeterogeneous(size_t childIndex, const std::string &paramName,
                                        const std::vector<std::vector<T>> &sortedGroupChildren, G getParamValuesFn) const
    {
        // Get value of archetype derived parameter
        const double firstValue = getParamValuesFn(sortedGroupChildren[0][childIndex]).at(paramName);

        // Loop through groups within merged group
        for(size_t i = 0; i < sortedGroupChildren.size(); i++) {
            const auto group = sortedGroupChildren[i][childIndex];
            if(getParamValuesFn(group).at(paramName) != firstValue) {
                return true;
            }
        }
       
        return false;
    }

    template<typename T = NeuronGroupMergedBase, typename C, typename H, typename V>
    void addHeterogeneousChildParams(const Snippet::Base::StringVec &paramNames,
                                     const std::vector<std::vector<C>> &sortedGroupChildren,
                                     size_t childIndex, const std::string &prefix,
                                     H isChildParamHeterogeneousFn, V getValueFn)
    {
        // Loop through parameters
        for(const auto &p : paramNames) {
            // If parameter is heterogeneous
            if((static_cast<const T*>(this)->*isChildParamHeterogeneousFn)(childIndex, p)) {
                addScalarField(p + prefix + std::to_string(childIndex),
                               [&sortedGroupChildren, childIndex, p, getValueFn](const NeuronGroupInternal &, size_t groupIndex)
                               {
                                   const auto *child = sortedGroupChildren.at(groupIndex).at(childIndex);
                                   return Utils::writePreciseString((child->*getValueFn)().at(p));
                               });
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename C, typename H, typename V>
    void addHeterogeneousChildDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams,
                                            const std::vector<std::vector<C>> &sortedGroupChildren,
                                            size_t childIndex, const std::string &prefix,
                                            H isChildDerivedParamHeterogeneousFn, V getValueFn)
    {
        // Loop through derived parameters
        for(const auto &p : derivedParams) {
            // If parameter is heterogeneous
            if((static_cast<const T*>(this)->*isChildDerivedParamHeterogeneousFn)(childIndex, p.name)) {
                addScalarField(p.name + prefix + std::to_string(childIndex),
                               [&sortedGroupChildren, childIndex, p, getValueFn](const NeuronGroupInternal &, size_t groupIndex)
                               {
                                   const auto *child = sortedGroupChildren.at(groupIndex).at(childIndex);
                                   return Utils::writePreciseString((child->*getValueFn)().at(p.name));
                               });
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename C, typename H, typename V>
    void addHeterogeneousChildVarInitParams(const Snippet::Base::StringVec &paramNames, 
                                            const std::vector<std::vector<C>> &sortedGroupChildren,
                                            size_t childIndex, const std::string &varName, const std::string &prefix,
                                            H isChildParamHeterogeneousFn, V getVarInitialiserFn)
    {
        // Loop through parameters
        for(const auto &p : paramNames) {
            // If parameter is heterogeneous
            if((static_cast<const T*>(this)->*isChildParamHeterogeneousFn)(childIndex, varName, p)) {
                addScalarField(p + varName + prefix + std::to_string(childIndex),
                               [&sortedGroupChildren, childIndex, varName, p, getVarInitialiserFn](const NeuronGroupInternal &, size_t groupIndex)
                               {
                                   const auto *child = sortedGroupChildren.at(groupIndex).at(childIndex);
                                   const auto &varInit = (child->*getVarInitialiserFn)();
                                   return Utils::writePreciseString(varInit.at(varName).getParams().at(p));
                               });
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename C, typename H, typename V>
    void addHeterogeneousChildVarInitDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, 
                                                   const std::vector<std::vector<C>> &sortedGroupChildren,
                                                   size_t childIndex, const std::string &varName, const std::string &prefix,
                                                   H isChildDerivedParamHeterogeneousFn, V getVarInitialiserFn)
    {
        // Loop through parameters
        for(const auto &d : derivedParams) {
            // If parameter is heterogeneous
            if((static_cast<const T*>(this)->*isChildDerivedParamHeterogeneousFn)(childIndex, varName, d.name)) {
                addScalarField(d.name + varName + prefix + std::to_string(childIndex),
                               [&sortedGroupChildren, childIndex, varName, d, getVarInitialiserFn](const NeuronGroupInternal &, size_t groupIndex)
                               {
                                   const auto *child = sortedGroupChildren.at(groupIndex).at(childIndex);
                                   const auto &varInit = (child->*getVarInitialiserFn)();
                                   return Utils::writePreciseString(varInit.at(varName).getDerivedParams().at(d.name));
                               });
            }
        }
    }

    template<typename S>
    void addChildEGPs(const std::vector<Snippet::Base::EGP> &egps, size_t childIndex,
                      const std::string &arrayPrefix, const std::string &prefix,
                      S getEGPSuffixFn)
    {
        for(const auto &e : egps) {
            const bool isPointer = Utils::isTypePointer(e.type);
            const std::string varPrefix = isPointer ? arrayPrefix : "";
            addField(e.type, e.name + prefix + std::to_string(childIndex),
                     [getEGPSuffixFn, childIndex, e, varPrefix](const NeuronGroupInternal&, size_t groupIndex)
                     {
                         return varPrefix + e.name + getEGPSuffixFn(groupIndex, childIndex);
                     },
                     Utils::isTypePointer(e.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
        }
    }

    template<typename T = NeuronGroupMergedBase, typename C, typename V, typename R>
    void updateChildParamHash(const std::vector<std::vector<C>> &sortedGroupChildren,
                              size_t childIndex, R isChildParamReferencedFn, V getValueFn, 
                              boost::uuids::detail::sha1 &hash) const
    {
        // Loop through parameters
        const auto &archetypeParams = (sortedGroupChildren.front().at(childIndex)->*getValueFn)();
        for(const auto &p : archetypeParams) {
            // If any of the code strings reference the parameter
            if((static_cast<const T*>(this)->*isChildParamReferencedFn)(childIndex, p.first)) {
                // Loop through groups
                for(size_t g = 0; g < getGroups().size(); g++) {
                    // Get child group
                    const auto *child = sortedGroupChildren.at(g).at(childIndex);

                    // Update hash with parameter value
                    Utils::updateHash((child->*getValueFn)().at(p.first), hash);
                }
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename C, typename V, typename R>
    void updateChildDerivedParamHash(const std::vector<std::vector<C>> &sortedGroupChildren,
                                     size_t childIndex,  R isChildDerivedParamReferencedFn, V getValueFn, 
                                     boost::uuids::detail::sha1 &hash) const
    {
        // Loop through derived parameters
        const auto &archetypeDerivedParams = (sortedGroupChildren.front().at(childIndex)->*getValueFn)();
        for(const auto &d : archetypeDerivedParams) {
            // If any of the code strings reference the parameter
            if((static_cast<const T*>(this)->*isChildDerivedParamReferencedFn)(childIndex, d.first)) {
                // Loop through groups
                for(size_t g = 0; g < getGroups().size(); g++) {
                    // Get child group
                    const auto *child = sortedGroupChildren.at(g).at(childIndex);

                    // Update hash with parameter value
                    Utils::updateHash((child->*getValueFn)().at(d.first), hash);
                }
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename C, typename R, typename V>
    void updateChildVarInitParamsHash(const std::vector<std::vector<C>> &sortedGroupChildren,
                                      size_t childIndex, const std::string &varName, R isChildParamReferencedFn, V getVarInitialiserFn,
                                      boost::uuids::detail::sha1 &hash) const
    {
        // Loop through parameters
        const auto &archetypeVarInit = (sortedGroupChildren.front().at(childIndex)->*getVarInitialiserFn)();
        const auto &archetypeParams = archetypeVarInit.at(varName).getParams();
        for(const auto &p : archetypeParams) {
            // If parameter is referenced
            if((static_cast<const T*>(this)->*isChildParamReferencedFn)(childIndex, varName, p.first)) {
                // Loop through groups
                for(size_t g = 0; g < getGroups().size(); g++) {
                    // Get child group and its variable initialisers
                    const auto *child = sortedGroupChildren.at(g).at(childIndex);
                    const auto &varInit = (child->*getVarInitialiserFn)();

                    // Update hash with parameter value
                    Utils::updateHash(varInit.at(varName).getParams().at(p.first), hash);
                }
            }
        }
    }

    template<typename T = NeuronGroupMergedBase, typename C, typename R, typename V>
    void updateChildVarInitDerivedParamsHash(const std::vector<std::vector<C>> &sortedGroupChildren,
                                             size_t childIndex, const std::string &varName, R isChildDerivedParamReferencedFn, V getVarInitialiserFn,
                                             boost::uuids::detail::sha1 &hash) const
    {
        // Loop through derived parameters
        const auto &archetypeVarInit = (sortedGroupChildren.front().at(childIndex)->*getVarInitialiserFn)();
        const auto &archetypeDerivedParams = archetypeVarInit.at(varName).getDerivedParams();
        for(const auto &d : archetypeDerivedParams) {
            // If parameter is referenced
            if((static_cast<const T*>(this)->*isChildDerivedParamReferencedFn)(childIndex, varName, d.first)) {
                // Loop through groups
                for(size_t g = 0; g < getGroups().size(); g++) {
                    // Get child group and its variable initialisers
                    const auto *child = sortedGroupChildren.at(g).at(childIndex);
                    const auto &varInit = (child->*getVarInitialiserFn)();

                    // Update hash with parameter value
                    Utils::updateHash(varInit.at(varName).getDerivedParams().at(d.first), hash);
                }
            }
        }
    }

    void addMergedInSynPointerField(const std::string &type, const std::string &name,
                                    size_t archetypeIndex, const std::string &prefix);

    void addMergedPreOutputOutSynPointerField(const std::string &type, const std::string &name,
                                    size_t archetypeIndex, const std::string &prefix);


private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<SynapseGroupInternal*>> m_SortedMergedInSyns;
    std::vector<std::vector<SynapseGroupInternal*>> m_SortedMergedPreOutputOutSyns;
    std::vector<std::vector<CurrentSourceInternal*>> m_SortedCurrentSources;
};



//----------------------------------------------------------------------------
// CodeGenerator::SynapseDendriticDelayUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDendriticDelayUpdateGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseDendriticDelayUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                           const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &group);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name);
    }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// SynapseConnectivityHostInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityHostInitGroupMerged : public GroupMerged<SynapseGroupInternal>
{
public:
    SynapseConnectivityHostInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                           const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

    //! Should the connectivity initialization parameter be implemented heterogeneously for EGP init?
    bool isConnectivityInitParamHeterogeneous(const std::string &paramName) const;

    //! Should the connectivity initialization derived parameter be implemented heterogeneously for EGP init?
    bool isConnectivityInitDerivedParamHeterogeneous(const std::string &paramName) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
     //! Is the connectivity initialization parameter referenced?
    bool isSparseConnectivityInitParamReferenced(const std::string &paramName) const;

    //! Is the connectivity initialization derived parameter referenced?
    bool isSparseConnectivityInitDerivedParamReferenced(const std::string &paramName) const;
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
    //! Should the weight update model parameter be implemented heterogeneously?
    bool isWUParamHeterogeneous(const std::string &paramName) const;

    //! Should the weight update model derived parameter be implemented heterogeneously?
    bool isWUDerivedParamHeterogeneous(const std::string &paramName) const;

    //! Should the GLOBALG weight update model variable be implemented heterogeneously?
    bool isWUGlobalVarHeterogeneous(const std::string &varName) const;

    //! Should the weight update model variable initialization parameter be implemented heterogeneously?
    bool isWUVarInitParamHeterogeneous(const std::string &varName, const std::string &paramName) const;
    
    //! Should the weight update model variable initialization derived parameter be implemented heterogeneously?
    bool isWUVarInitDerivedParamHeterogeneous(const std::string &varName, const std::string &paramName) const;

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
    bool isKernelSizeHeterogeneous(size_t dimensionIndex) const;
    
    //! Get expression for kernel size in dimension (may be literal or group->kernelSizeXXX)
    std::string getKernelSize(size_t dimensionIndex) const;
    
    //! Generate an index into a kernel based on the id_kernel_XXX variables in subs
    void genKernelIndex(std::ostream &os, const CodeGenerator::Substitutions &subs) const;

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

    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    static std::string getPreVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    static std::string getPostVarIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);

    static std::string getPrePrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    static std::string getPostPrevSpikeTimeIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    
    static std::string getPostISynIndex(unsigned int batchSize, const std::string &index)
    {
        return ((batchSize == 1) ? "" : "postBatchOffset + ") + index;
    }

    static std::string getPreISynIndex(unsigned int batchSize, const std::string &index)
    {
        return ((batchSize == 1) ? "" : "preBatchOffset + ") + index;
    }

    static std::string getSynVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    static std::string getKernelVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    
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
        KernelInit,
        ConnectivityInit,
    };

    SynapseGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                           Role role, const std::string &archetypeCode, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    boost::uuids::detail::sha1::digest_type getHashDigest(Role role) const;

    const std::string &getArchetypeCode() const { return m_ArchetypeCode; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void addPSPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addPreOutputPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addSrcPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addTrgPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addWeightSharingPointerField(const std::string &type, const std::string &name, const std::string &prefix);

    //! Is the weight update model parameter referenced?
    bool isWUParamReferenced(const std::string &paramName) const;

    //! Is the weight update model derived parameter referenced?
    bool isWUDerivedParamReferenced(const std::string &paramName) const;

    //! Is the GLOBALG weight update model variable referenced?
    bool isWUGlobalVarReferenced(const std::string &varName) const;

    //! Is the weight update model variable initialization parameter referenced?
    bool isWUVarInitParamReferenced(const std::string &varName, const std::string &paramName) const;
    
    //! Is the weight update model variable initialization derived parameter referenced?
    bool isWUVarInitDerivedParamReferenced(const std::string &varName, const std::string &paramName) const;

    //! Is the sparse connectivity initialization parameter referenced?
    bool isSparseConnectivityInitParamReferenced(const std::string &paramName) const;

    //! Is the sparse connectivity initialization parameter referenced?
    bool isSparseConnectivityInitDerivedParamReferenced(const std::string &paramName) const;

    //! Is the toeplitz connectivity initialization parameter referenced?
    bool isToeplitzConnectivityInitParamReferenced(const std::string &paramName) const;

    //! Is the toeplitz connectivity initialization parameter referenced?
    bool isToeplitzConnectivityInitDerivedParamReferenced(const std::string &paramName) const;

    //! Is presynaptic neuron parameter referenced?
    bool isSrcNeuronParamReferenced(const std::string &paramName) const;

    //! Is presynaptic neuron derived parameter referenced?
    bool isSrcNeuronDerivedParamReferenced(const std::string &paramName) const;

    //! Is postsynaptic neuron parameter referenced?
    bool isTrgNeuronParamReferenced(const std::string &paramName) const;

    //! Is postsynaptic neuron derived parameter referenced?
    bool isTrgNeuronDerivedParamReferenced(const std::string &paramName) const;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_ArchetypeCode;
};

// ----------------------------------------------------------------------------
// CustomUpdateHostReductionGroupMergedBase
//----------------------------------------------------------------------------
template<typename G>
class CustomUpdateHostReductionGroupMergedBase : public GroupMerged<G>
{
protected:
     CustomUpdateHostReductionGroupMergedBase(size_t index, const std::string &precision, const BackendBase &backend,
                                   const std::vector<std::reference_wrapper<const G>> &groups)
    :   GroupMerged<G>(index, precision, groups)
    {
        // Loop through variables and add pointers if they are reduction targets
        const CustomUpdateModels::Base *cm = this->getArchetype().getCustomUpdateModel();
        for(const auto &v : cm->getVars()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                this->addPointerField(v.type, v.name, backend.getDeviceVarPrefix() + v.name);
            }
        }

        // Loop through variable references and add pointers if they are reduction targets
        for(const auto &v : cm->getVarRefs()) {
            if(v.access & VarAccessModeAttribute::REDUCE) {
                this->addPointerField(v.type, v.name, backend.getDeviceVarPrefix() + v.name);
            }
        }
    }
};

// ----------------------------------------------------------------------------
// CustomUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateHostReductionGroupMerged : public CustomUpdateHostReductionGroupMergedBase<CustomUpdateInternal>
{
public:
    CustomUpdateHostReductionGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                         const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// CustomWUUpdateHostReductionGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateHostReductionGroupMerged : public CustomUpdateHostReductionGroupMergedBase<CustomUpdateWUInternal>
{
public:
    CustomWUUpdateHostReductionGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                           const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void generateRunner(const BackendBase &backend, CodeStream &definitionsInternal,
                        CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc) const
    {
        generateRunnerBase(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                           runnerVarDecl, runnerMergedStructAlloc, name, true);
    }

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};
}   // namespace CodeGenerator
