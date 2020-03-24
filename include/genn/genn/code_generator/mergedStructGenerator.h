#pragma once

// Standard C++ includes
#include <functional>
#include <vector>
#include <unordered_map>

// GeNN includes
#include "gennUtils.h"

// GeNN code generator includes
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

    void addVars(const std::vector<Models::Base::Var> &vars, const std::string &prefix)
    {
        for(const auto &v : vars) {
            addPointerField(v.type, v.name, prefix + v.name);
        }
    }

    void addEGPs(const std::vector<Snippet::Base::EGP> &egps, const std::string &prefix)
    {
        for(const auto &e : egps) {
            addField(e.type, e.name,
                     [e, prefix](const typename T::GroupInternal &g, size_t){ return prefix + e.name + g.getName(); },
                     Utils::isTypePointer(e.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
        }
    }

    void addEGPPointers(const std::vector<Snippet::Base::EGP> &egps, const std::string &prefix)
    {
        for(const auto &e : egps) {
            addField(e.type + "*", prefix + e.name,
                     [e, prefix](const typename T::GroupInternal &g, size_t) { return "&" + prefix + e.name + g.getName(); });
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

    void generate(const BackendBase &backend, CodeStream &definitionsInternal, 
                  CodeStream &definitionsInternalFunc, CodeStream &definitionsInternalVar, 
                  CodeStream &runnerVarDecl, CodeStream &runnerVarAlloc, 
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

        // Write struct declation to top of definitions internal
        size_t structSize = 0;
        size_t largestFieldSize = 0;
        definitionsInternal << "struct Merged" << name << "Group" << mergedGroupIndex << std::endl;
        {
            CodeStream::Scope b(definitionsInternal);
            for(const auto &f : sortedFields) {
                // Add field to structure
                definitionsInternal << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;

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
            definitionsInternal << std::endl;
        }

        definitionsInternal << ";" << std::endl;

        // Add total size of array of merged structures to merged struct data
        // **NOTE** to match standard struct packing rules we pad to a multiple of the largest field size
        const size_t arraySize = padSize(structSize, largestFieldSize) * getMergedGroup().getGroups().size();
        mergedStructData.addMergedGroupSize(name, mergedGroupIndex, arraySize);

        // Declare array of these structs containing individual neuron group pointers etc
        runnerVarDecl << "Merged" << name << "Group" << mergedGroupIndex << " merged" << name << "Group" << mergedGroupIndex << "[" << getMergedGroup().getGroups().size() << "];" << std::endl;

        for(size_t groupIndex = 0; groupIndex < getMergedGroup().getGroups().size(); groupIndex++) {
            const auto &g = getMergedGroup().getGroups()[groupIndex];

            // Set all fields in array of structs
            runnerVarAlloc << "merged" << name << "Group" << mergedGroupIndex << "[" << groupIndex << "] = {";
            for(const auto &f : sortedFields) {
                const std::string fieldInitVal = std::get<2>(f)(g, groupIndex);
                runnerVarAlloc << fieldInitVal << ", ";

                // If field is an EGP, add record to merged EGPS
                if(std::get<3>(f) != FieldType::Standard) {
                    mergedStructData.addMergedEGP(fieldInitVal, name, mergedGroupIndex, groupIndex,
                                                  std::get<0>(f), std::get<1>(f));
                }
            }
            runnerVarAlloc << "};" << std::endl;

        }

        // If this is a host merged struct, export the variable
        if(host) {
            definitionsInternalVar << "EXPORT_VAR Merged" << name << "Group" << mergedGroupIndex << " merged" << name << "Group" << mergedGroupIndex << "[" << getMergedGroup().getGroups().size() << "]; " << std::endl;
        }
        // Otherwise
        else {
            // Then generate call to function to copy local array to device
            runnerVarAlloc << "pushMerged" << name << "Group" << mergedGroupIndex << "ToDevice(merged" << name << "Group" << mergedGroupIndex << ");" << std::endl;

            // Finally add declaration to function to definitions internal
            definitionsInternalFunc << "EXPORT_FUNC void pushMerged" << name << "Group" << mergedGroupIndex << "ToDevice(const Merged" << name << "Group" << mergedGroupIndex << " *group);" << std::endl;
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

//--------------------------------------------------------------------------
// CodeGenerator::MergedNeuronStructGenerator
//--------------------------------------------------------------------------
class MergedNeuronStructGenerator : public MergedStructGenerator<CodeGenerator::NeuronGroupMerged>
{
public:
    MergedNeuronStructGenerator(const CodeGenerator::NeuronGroupMerged &mergedGroup, const std::string &precision)
    :   MergedStructGenerator<CodeGenerator::NeuronGroupMerged>(mergedGroup, precision)
    {
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addMergedInSynPointerField(const std::string &type, const std::string &name, size_t archetypeIndex, const std::string &prefix,
                                    const std::vector<std::vector<std::pair<SynapseGroupInternal*, std::vector<SynapseGroupInternal*>>>> &sortedMergedInSyns)
    {
        assert(!Utils::isTypePointer(type));
        addField(type + "*", name + std::to_string(archetypeIndex),
                 [prefix, &sortedMergedInSyns, archetypeIndex](const NeuronGroupInternal&, size_t groupIndex)
                 {
                     return prefix + sortedMergedInSyns.at(groupIndex).at(archetypeIndex).first->getPSModelTargetName();
                 });
    }

    void addCurrentSourcePointerField(const std::string &type, const std::string &name, size_t archetypeIndex, const std::string &prefix,
                                      const std::vector<std::vector<CurrentSourceInternal*>> &sortedCurrentSources)
    {
        assert(!Utils::isTypePointer(type));
        addField(type + "*", name + std::to_string(archetypeIndex),
                 [prefix, &sortedCurrentSources, archetypeIndex](const NeuronGroupInternal&, size_t groupIndex)
                 {
                     return prefix + sortedCurrentSources.at(groupIndex).at(archetypeIndex)->getName();
                 });
    }

    void addSynPointerField(const std::string &type, const std::string &name, size_t archetypeIndex, const std::string &prefix,
                            const std::vector<std::vector<SynapseGroupInternal*>> &sortedSyn)
    {
        assert(!Utils::isTypePointer(type));
        addField(type + "*", name + std::to_string(archetypeIndex),
                 [prefix, &sortedSyn, archetypeIndex](const NeuronGroupInternal&, size_t groupIndex)
                 {
                     return prefix + sortedSyn.at(groupIndex).at(archetypeIndex)->getName();
                 });

    }
};
//--------------------------------------------------------------------------
// CodeGenerator::MergedSynapseStructGenerator
//--------------------------------------------------------------------------
class MergedSynapseStructGenerator : public MergedStructGenerator<CodeGenerator::SynapseGroupMerged>
{
public:
    MergedSynapseStructGenerator(const CodeGenerator::SynapseGroupMerged &mergedGroup, const std::string &precision)
    :   MergedStructGenerator<CodeGenerator::SynapseGroupMerged>(mergedGroup, precision)
    {
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addPSPointerField(const std::string &type, const std::string &name, const std::string &prefix)
    {
        assert(!Utils::isTypePointer(type));
        addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t){ return prefix + sg.getPSModelTargetName(); });
    }

    void addSrcPointerField(const std::string &type, const std::string &name, const std::string &prefix)
    {
        assert(!Utils::isTypePointer(type));
        addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t){ return prefix + sg.getSrcNeuronGroup()->getName(); });
    }

    void addTrgPointerField(const std::string &type, const std::string &name, const std::string &prefix)
    {
        assert(!Utils::isTypePointer(type));
        addField(type + "*", name, [prefix](const SynapseGroupInternal &sg, size_t){ return prefix + sg.getTrgNeuronGroup()->getName(); });
    }

    void addSrcEGPField(const Snippet::Base::EGP &egp)
    {
        addField(egp.type, egp.name + "Pre",
                 [egp](const SynapseGroupInternal &sg, size_t){ return egp.name + sg.getSrcNeuronGroup()->getName(); },
                 Utils::isTypePointer(egp.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
    }

    void addTrgEGPField(const Snippet::Base::EGP &egp)
    {
        addField(egp.type, egp.name + "Post",
                 [egp](const SynapseGroupInternal &sg, size_t){ return egp.name + sg.getTrgNeuronGroup()->getName(); },
                 Utils::isTypePointer(egp.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
    }

};

}   // namespace CodeGenerator
