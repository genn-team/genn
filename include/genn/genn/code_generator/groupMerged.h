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
    bool isParamValueHeterogeneous(const std::vector<std::string> &codeStrings, const std::string &paramName,
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
        for(size_t v = 0; v < varReferences.size(); v++) {
            addField(varReferences[v].type + "*", varReferences[v].name, 
                     [getVarRefFn, arrayPrefix, v](const G &g, size_t) 
                     { 
                         const auto varRef = getVarRefFn(g).at(v);
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
        for(size_t p = 0; p < paramNames.size(); p++) {
            // If parameters is heterogeneous
            if((static_cast<const T*>(this)->*isHeterogeneous)(p)) {
                // Add field
                addScalarField(paramNames[p] + suffix,
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
        for(size_t p = 0; p < derivedParams.size(); p++) {
            // If parameters isn't homogeneous
            if((static_cast<const T*>(this)->*isHeterogeneous)(p)) {
                // Add field
                addScalarField(derivedParams[p].name + suffix,
                               [p, getDerivedParamValues](const G &g, size_t)
                               {
                                   const auto &values = getDerivedParamValues(g);
                                   return Utils::writePreciseString(values.at(p));
                               });
            }
        }
    }

    template<typename T, typename V, typename H>
    void addHeterogeneousVarInitParams(const Models::Base::VarVec &vars, V getVarInitialisers, H isHeterogeneous)
    {
        // Loop through weight update model variables
        const std::vector<Models::VarInit> &archetypeVarInitialisers = (getArchetype().*getVarInitialisers)();
        for(size_t v = 0; v < archetypeVarInitialisers.size(); v++) {
            // Loop through parameters
            const Models::VarInit &varInit = archetypeVarInitialisers[v];
            for(size_t p = 0; p < varInit.getParams().size(); p++) {
                if((static_cast<const T*>(this)->*isHeterogeneous)(v, p)) {
                    addScalarField(varInit.getSnippet()->getParamNames()[p] + vars[v].name,
                                   [p, v, getVarInitialisers](const G &g, size_t)
                                   {
                                       const auto &values = (g.*getVarInitialisers)()[v].getParams();
                                       return Utils::writePreciseString(values.at(p));
                                   });
                }
            }
        }
    }

    template<typename T, typename V, typename H>
    void addHeterogeneousVarInitDerivedParams(const Models::Base::VarVec &vars, V getVarInitialisers, H isHeterogeneous)
    {
        // Loop through weight update model variables
        const std::vector<Models::VarInit> &archetypeVarInitialisers = (getArchetype().*getVarInitialisers)();
        for(size_t v = 0; v < archetypeVarInitialisers.size(); v++) {
            // Loop through parameters
            const Models::VarInit &varInit = archetypeVarInitialisers[v];
            for(size_t d = 0; d < varInit.getDerivedParams().size(); d++) {
                if((static_cast<const T*>(this)->*isHeterogeneous)(v, d)) {
                    addScalarField(varInit.getSnippet()->getDerivedParams()[d].name + vars[v].name,
                                   [d, v, getVarInitialisers](const G &g, size_t)
                                   {
                                       const auto &values = (g.*getVarInitialisers)()[v].getDerivedParams();
                                       return Utils::writePreciseString(values.at(d));
                                   });
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
    bool isChildParamValueHeterogeneous(const std::vector<std::string> &codeStrings,
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
            if((static_cast<const T*>(this)->*isChildParamHeterogeneousFn)(childIndex, p)) {
                addScalarField(paramNames[p] + prefix + std::to_string(childIndex),
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
            if((static_cast<const T*>(this)->*isChildDerivedParamHeterogeneousFn)(childIndex, p)) {
                addScalarField(derivedParams[p].name + prefix + std::to_string(childIndex),
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
                addScalarField(paramNames[p] + prefix + std::to_string(childIndex),
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
            if((static_cast<const T*>(this)->*isChildDerivedParamHeterogeneousFn)(childIndex, varIndex, p)) {
                addScalarField(derivedParams[p].name + prefix + std::to_string(childIndex),
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
    


    void addMergedInSynPointerField(const std::string &type, const std::string &name, 
                                    size_t archetypeIndex, const std::string &prefix);

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::vector<std::vector<SynapseGroupInternal*>> m_SortedMergedInSyns;
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
    //! Should the incoming synapse weight update model parameter be implemented heterogeneously?
    bool isInSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the incoming synapse weight update model derived parameter be implemented heterogeneously?
    bool isInSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model parameter be implemented heterogeneously?
    bool isOutSynWUMParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

    //! Should the outgoing synapse weight update model derived parameter be implemented heterogeneously?
    bool isOutSynWUMDerivedParamHeterogeneous(size_t childIndex, size_t paramIndex) const;

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
    bool isConnectivityInitParamHeterogeneous(size_t paramIndex) const;

    //! Should the connectivity initialization derived parameter be implemented heterogeneously for EGP init?
    bool isConnectivityInitDerivedParamHeterogeneous(size_t paramIndex) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
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

    //! Is presynaptic neuron parameter heterogeneous?
    bool isSrcNeuronParamHeterogeneous(size_t paramIndex) const;

    //! Is presynaptic neuron derived parameter heterogeneous?
    bool isSrcNeuronDerivedParamHeterogeneous(size_t paramIndex) const;

    //! Is postsynaptic neuron parameter heterogeneous?
    bool isTrgNeuronParamHeterogeneous(size_t paramIndex) const;

    //! Is postsynaptic neuron derived parameter heterogeneous?
    bool isTrgNeuronDerivedParamHeterogeneous(size_t paramIndex) const;

    //! Is kernel size heterogeneous in this dimension?
    bool isKernelSizeHeterogeneous(size_t dimensionIndex) const;

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

    static std::string getSynVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    
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
        ConnectivityInit,
    };

    SynapseGroupMergedBase(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                           Role role, const std::string &archetypeCode, const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups);

    //----------------------------------------------------------------------------
    // Protected methods
    //----------------------------------------------------------------------------
    const std::string &getArchetypeCode() const { return m_ArchetypeCode; }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void addPSPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addSrcPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addTrgPointerField(const std::string &type, const std::string &name, const std::string &prefix);
    void addWeightSharingPointerField(const std::string &type, const std::string &name, const std::string &prefix);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const std::string m_ArchetypeCode;
};

//----------------------------------------------------------------------------
// CodeGenerator::PresynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PresynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    PresynapticUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::PresynapticUpdate, 
                               groups.front().get().getWUModel()->getSimCode() + groups.front().get().getWUModel()->getEventCode() + groups.front().get().getWUModel()->getEventThresholdConditionCode(), groups)
    {}

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
// CodeGenerator::PostsynapticUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT PostsynapticUpdateGroupMerged : public SynapseGroupMergedBase
{
public:
    PostsynapticUpdateGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                  const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::PostsynapticUpdate, 
                               groups.front().get().getWUModel()->getLearnPostCode(), groups)
    {}

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
// CodeGenerator::SynapseDynamicsGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDynamicsGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseDynamicsGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                               const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::SynapseDynamics, 
                               groups.front().get().getWUModel()->getSynapseDynamicsCode(), groups)
    {}

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
// CodeGenerator::SynapseDenseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseDenseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseDenseInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::DenseInit, "", groups)
    {}

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
// CodeGenerator::SynapseSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseSparseInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseSparseInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend, 
                                 const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::SparseInit, "", groups)
    {}

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
// SynapseConnectivityInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT SynapseConnectivityInitGroupMerged : public SynapseGroupMergedBase
{
public:
    SynapseConnectivityInitGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const SynapseGroupInternal>> &groups)
    :   SynapseGroupMergedBase(index, precision, timePrecision, backend, SynapseGroupMergedBase::Role::ConnectivityInit, "", groups)
    {}

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
// CustomUpdateGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateGroupMerged : public GroupMerged<CustomUpdateInternal>
{
public:
    CustomUpdateGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                            const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(size_t index) const;
    bool isDerivedParamHeterogeneous(size_t index) const;

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
    std::string getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;
    std::string getVarRefIndex(bool delay, unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index) const;

    //----------------------------------------------------------------------------
    // Static constants
    //----------------------------------------------------------------------------
    static const std::string name;
};

// ----------------------------------------------------------------------------
// CustomUpdateWUGroupMergedBase
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWUGroupMergedBase : public GroupMerged<CustomUpdateWUInternal>
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool isParamHeterogeneous(size_t index) const;
    bool isDerivedParamHeterogeneous(size_t index) const;

    //----------------------------------------------------------------------------
    // Static API
    //----------------------------------------------------------------------------
    static std::string getVarIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);
    static std::string getVarRefIndex(unsigned int batchSize, VarAccessDuplication varDuplication, const std::string &index);

protected:
    CustomUpdateWUGroupMergedBase(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                  const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);
};

// ----------------------------------------------------------------------------
// CustomUpdateWUGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateWUGroupMerged : public CustomUpdateWUGroupMergedBase
{
public:
    CustomUpdateWUGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                              const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
    :   CustomUpdateWUGroupMergedBase(index, precision, timePrecision, backend, groups)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
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
// CustomUpdateTransposeWUGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateTransposeWUGroupMerged : public CustomUpdateWUGroupMergedBase
{
public:
    CustomUpdateTransposeWUGroupMerged(size_t index, const std::string &precision, const std::string &timePrecision, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups)
    :   CustomUpdateWUGroupMergedBase(index, precision, timePrecision, backend, groups)
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
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
// CustomUpdateInitGroupMergedBase
//----------------------------------------------------------------------------
template<typename G>
class CustomUpdateInitGroupMergedBase : public GroupMerged<G>
{
public:
     //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
     //! Should the var init parameter be implemented heterogeneously?
    bool isVarInitParamHeterogeneous(size_t varIndex, size_t paramIndex) const
    {
        // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
        const auto *varInitSnippet = this->getArchetype().getVarInitialisers().at(varIndex).getSnippet();
        const std::string paramName = varInitSnippet->getParamNames().at(paramIndex);
        return this->isParamValueHeterogeneous({varInitSnippet->getCode()}, paramName, paramIndex,
                                               [varIndex](const G &cg)
                                               {
                                                   return cg.getVarInitialisers().at(varIndex).getParams();
                                               });
    }

    //! Should the var init derived parameter be implemented heterogeneously?
    bool isVarInitDerivedParamHeterogeneous(size_t varIndex, size_t paramIndex) const
    {
        // If parameter isn't referenced in code, there's no point implementing it hetereogeneously!
        const auto *varInitSnippet = this->getArchetype().getVarInitialisers().at(varIndex).getSnippet();
        const std::string derivedParamName = varInitSnippet->getDerivedParams().at(paramIndex).name;
        return this->isParamValueHeterogeneous({varInitSnippet->getCode()}, derivedParamName, paramIndex,
                                               [varIndex](const G &cg)
                                               {
                                                   return cg.getVarInitialisers().at(varIndex).getDerivedParams();
                                               });
    }

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

        this->template  addHeterogeneousVarInitParams<CustomUpdateInitGroupMergedBase<G>>(
            vars, &G::getVarInitialisers,
            &CustomUpdateInitGroupMergedBase<G>::isVarInitParamHeterogeneous);

        this->template addHeterogeneousVarInitDerivedParams<CustomUpdateInitGroupMergedBase<G>>(
            vars, &G::getVarInitialisers,
            &CustomUpdateInitGroupMergedBase<G>::isVarInitDerivedParamHeterogeneous);
    }
};

// ----------------------------------------------------------------------------
// CustomUpdateInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomUpdateInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateInternal>
{
public:
    CustomUpdateInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                const std::vector<std::reference_wrapper<const CustomUpdateInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
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
// CustomWUUpdateDenseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateDenseInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal>
{
public:
    CustomWUUpdateDenseInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                       const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
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
// CustomWUUpdateSparseInitGroupMerged
//----------------------------------------------------------------------------
class GENN_EXPORT CustomWUUpdateSparseInitGroupMerged : public CustomUpdateInitGroupMergedBase<CustomUpdateWUInternal>
{
public:
    CustomWUUpdateSparseInitGroupMerged(size_t index, const std::string &precision, const std::string &, const BackendBase &backend,
                                        const std::vector<std::reference_wrapper<const CustomUpdateWUInternal>> &groups);

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
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
}   // namespace CodeGenerator
