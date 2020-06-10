#pragma once

// Standard C++ includes
#include <functional>
#include <vector>
#include <unordered_map>

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/groupMerged.h"

//--------------------------------------------------------------------------
// CodeGenerator::MergedStructGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
template<typename T>
class MergedStructGenerator
{
public:
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class FieldType
    {
        Standard,
        ScalarEGP,
        PointerEGP,
    };

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef std::function<std::string(const typename T::GroupInternal &, size_t)> GetFieldValueFunc;
 
    MergedStructGenerator(const T &mergedGroup, const std::string &precision) : m_MergedGroup(mergedGroup), m_LiteralSuffix((precision == "float") ? "f" : "")
    {
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addField(const std::string &type, const std::string &name, GetFieldValueFunc getFieldValue, FieldType fieldType = FieldType::Standard)
    {
        m_Fields.emplace_back(type, name, getFieldValue, fieldType);
    }

    void addScalarField(const std::string &name, GetFieldValueFunc getFieldValue, FieldType fieldType = FieldType::Standard)
    {
        addField("scalar", name,
                 [getFieldValue, this](const typename T::GroupInternal &g, size_t i)
                 {
                    return getFieldValue(g, i) + m_LiteralSuffix;
                 },
                 fieldType);
    }

    void addPointerField(const std::string &type, const std::string &name, const std::string &prefix)
    {
        assert(!Utils::isTypePointer(type));
        addField(type + "*", name, [prefix](const typename T::GroupInternal &g, size_t){ return prefix + g.getName(); });
    }


    void addVars(const Models::Base::VarVec &vars, const std::string &arrayPrefix)
    {
        // Loop through variables
        for(const auto &v : vars) {
            addPointerField(v.type, v.name, arrayPrefix + v.name);

        }
    }

    void addEGPs(const Snippet::Base::EGPVec &egps, const std::string &arrayPrefix, const std::string &varName = "")
    {
        for(const auto &e : egps) {
            const bool isPointer = Utils::isTypePointer(e.type);
            const std::string prefix = isPointer ? arrayPrefix : "";
            addField(e.type, e.name + varName,
                     [e, prefix, varName](const typename T::GroupInternal &g, size_t){ return prefix + e.name + varName + g.getName(); },
                     isPointer ? FieldType::PointerEGP : FieldType::ScalarEGP);
        }
    }

    template<typename G, typename H>
    void addHeterogeneousParams(const Snippet::Base::StringVec &paramNames, 
                                G getParamValues, H isHeterogeneous)
    {
        // Loop through params
        for(size_t p = 0; p < paramNames.size(); p++) {
            // If parameters is heterogeneous
            if((getMergedGroup().*isHeterogeneous)(p)) {
                // Add field
                addScalarField(paramNames[p],
                               [p, getParamValues](const typename T::GroupInternal &g, size_t)
                               {
                                   const auto &values = getParamValues(g);
                                   return Utils::writePreciseString(values.at(p));
                               });
            }
        }
    }

    template<typename G, typename H>
    void addHeterogeneousDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, 
                                       G getDerivedParamValues, H isHeterogeneous)
    { 
        // Loop through derived params
        for(size_t p = 0; p < derivedParams.size(); p++) {
            // If parameters isn't homogeneous
            if((getMergedGroup().*isHeterogeneous)(p)) {
                // Add field
                addScalarField(derivedParams[p].name,
                               [p, getDerivedParamValues](const typename T::GroupInternal &g, size_t)
                               {
                                   const auto &values = getDerivedParamValues(g);
                                   return Utils::writePreciseString(values.at(p));
                               });
            }
        }
    }

    template<typename V, typename H>
    void addHeterogeneousVarInitParams(const Models::Base::VarVec &vars, V getVarInitialisers, H isHeterogeneous)
    {
        // Loop through weight update model variables
        const std::vector<Models::VarInit> &archetypeVarInitialisers = (getMergedGroup().getArchetype().*getVarInitialisers)();
        for(size_t v = 0; v < archetypeVarInitialisers.size(); v++) {
            // Loop through parameters
            const Models::VarInit &varInit = archetypeVarInitialisers[v];
            for(size_t p = 0; p < varInit.getParams().size(); p++) {
                if((getMergedGroup().*isHeterogeneous)(v, p)) {
                    addScalarField(varInit.getSnippet()->getParamNames()[p] + vars[v].name,
                                   [p, v, getVarInitialisers](const typename T::GroupInternal &g, size_t)
                                   {
                                       const auto &values = (g.*getVarInitialisers)()[v].getParams();
                                       return Utils::writePreciseString(values.at(p));
                                   });
                }
            }
        }
    }

    template<typename V, typename H>
    void addHeterogeneousVarInitDerivedParams(const Models::Base::VarVec &vars, V getVarInitialisers, H isHeterogeneous)
    {
        // Loop through weight update model variables
        const std::vector<Models::VarInit> &archetypeVarInitialisers = (getMergedGroup().getArchetype().*getVarInitialisers)();
        for(size_t v = 0; v < archetypeVarInitialisers.size(); v++) {
            // Loop through parameters
            const Models::VarInit &varInit = archetypeVarInitialisers[v];
            for(size_t d = 0; d < varInit.getDerivedParams().size(); d++) {
                if((getMergedGroup().*isHeterogeneous)(v, d)) {
                    addScalarField(varInit.getSnippet()->getDerivedParams()[d].name + vars[v].name,
                                   [d, v, getVarInitialisers](const typename T::GroupInternal &g, size_t)
                                   {
                                       const auto &values = (g.*getVarInitialisers)()[v].getDerivedParams();
                                       return Utils::writePreciseString(values.at(d));
                                   });
                }
            }
        }
    }

    //! Generate declaration of struct
    void generateStruct(const BackendBase &backend, CodeStream &os, 
                        const std::string &name, const std::string &prefix = "") const
    {
        // Make a copy of fields and sort so largest come first. This should mean that due
        // to structure packing rules, significant memory is saved and estimate is more precise
        auto sortedFields = m_Fields;
        std::sort(sortedFields.begin(), sortedFields.end(),
                  [&backend](const Field &a, const Field &b)
                  {
                      return (backend.getSize(std::get<0>(a)) > backend.getSize(std::get<0>(b)));
                  });

        os << "struct Merged" << name << "Group" << getMergedGroup().getIndex() << std::endl;
        {
            // Loop through fields and write to structure
            CodeStream::Scope b(os);
            for(const auto &f : sortedFields) {
                os << prefix << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;
            }
            os << std::endl;
        }

        os << ";" << std::endl;
    }

    void generate(const BackendBase &backend, CodeStream &definitionsInternal, 
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar, 
                  CodeStream &runnerVarDecl, CodeStream &runnerMergedStructAlloc,
                  MergedStructData &mergedStructData, const std::string &name, bool host = false) const
    {
        const size_t mergedGroupIndex = getMergedGroup().getIndex();

        // Make a copy of fields and sort so largest come first. This should mean that due
        // to structure packing rules, significant memory is saved and estimate is more precise
        auto sortedFields = m_Fields;
        std::sort(sortedFields.begin(), sortedFields.end(),
                  [&backend](const Field &a, const Field &b)
                  {
                      return (backend.getSize(std::get<0>(a)) > backend.getSize(std::get<0>(b)));
                  });

        // If this isn't a host merged structure, generate definition for function to push group
        if(!host) {
            definitionsInternalFunc << "EXPORT_FUNC void pushMerged" << name << "Group" << mergedGroupIndex << "ToDevice(unsigned int idx, ";
            for(size_t fieldIndex = 0; fieldIndex < sortedFields.size(); fieldIndex++) {
                const auto &f = sortedFields[fieldIndex];
                definitionsInternalFunc << backend.getMergedGroupFieldHostType(std::get<0>(f)) << " " << std::get<1>(f);
                if(fieldIndex != (sortedFields.size() - 1)) {
                    definitionsInternalFunc << ", ";
                }
            }
            definitionsInternalFunc << ");" << std::endl;
        }

        // Loop through fields again to generate any EGP pushing functions that are required and to calculate struct size
        size_t structSize = 0;
        size_t largestFieldSize = 0;
        for(const auto &f : sortedFields) {
            // Add size of field to total
            const size_t fieldSize = backend.getSize(std::get<0>(f));
            structSize += fieldSize;

            // Update largest field size
            largestFieldSize = std::max(fieldSize, largestFieldSize);

            // If this field is for a pointer EGP, also declare function to push it
            if(std::get<3>(f) == FieldType::PointerEGP) {
                definitionsInternalFunc << "EXPORT_FUNC void pushMerged" << name << mergedGroupIndex << std::get<1>(f) << "ToDevice(unsigned int idx, " << std::get<0>(f) << " value);" << std::endl;
            }
        }

        // Add total size of array of merged structures to merged struct data
        // **NOTE** to match standard struct packing rules we pad to a multiple of the largest field size
        const size_t arraySize = padSize(structSize, largestFieldSize) * getMergedGroup().getGroups().size();
        mergedStructData.addMergedGroupSize(name, mergedGroupIndex, arraySize);

        // If merged group is used on host
        if(host) {
            // Generate struct directly into internal definitions
            generateStruct(backend, definitionsInternal, name);

            // Declare array of these structs containing individual neuron group pointers etc
            runnerVarDecl << "Merged" << name << "Group" << mergedGroupIndex << " merged" << name << "Group" << mergedGroupIndex << "[" << getMergedGroup().getGroups().size() << "];" << std::endl;
            
            // Export it
            definitionsInternalVar << "EXPORT_VAR Merged" << name << "Group" << mergedGroupIndex << " merged" << name << "Group" << mergedGroupIndex << "[" << getMergedGroup().getGroups().size() << "]; " << std::endl;
        }

        // Loop through groups
        for(size_t groupIndex = 0; groupIndex < getMergedGroup().getGroups().size(); groupIndex++) {
            const auto &g = getMergedGroup().getGroups()[groupIndex];

            // If this is a merged group used on the host, directly set array entry
            if(host) {
                runnerMergedStructAlloc << "merged" << name << "Group" << mergedGroupIndex << "[" << groupIndex << "] = {";
            }
            // Otherwise, call function to push to device
            else {
                runnerMergedStructAlloc << "pushMerged" << name << "Group" << mergedGroupIndex << "ToDevice(" << groupIndex << ", ";
            }

            // Loop through fields
            for(size_t fieldIndex = 0; fieldIndex < sortedFields.size(); fieldIndex++) {
                const auto &f = sortedFields[fieldIndex];
                const std::string fieldInitVal = std::get<2>(f)(g, groupIndex);
                runnerMergedStructAlloc << fieldInitVal;
                if(fieldIndex != (sortedFields.size() - 1)) {
                    runnerMergedStructAlloc << ", ";
                }

                // If field is an EGP, add record to merged EGPS
                if(std::get<3>(f) != FieldType::Standard) {
                    mergedStructData.addMergedEGP(fieldInitVal, name, mergedGroupIndex, groupIndex,
                                                  std::get<0>(f), std::get<1>(f));
                }
            }

            if(host) {
                runnerMergedStructAlloc << "};" << std::endl;
            }
            else {
                runnerMergedStructAlloc << ");" << std::endl;
            }

        }
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const T &getMergedGroup() const{ return m_MergedGroup; }

private:
    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef std::tuple<std::string, std::string, GetFieldValueFunc, FieldType> Field;

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const T &m_MergedGroup;
    const std::string m_LiteralSuffix;
    std::vector<Field> m_Fields;
};
}   // namespace CodeGenerator
