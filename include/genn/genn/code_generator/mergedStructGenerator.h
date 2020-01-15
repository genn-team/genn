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
 
    MergedStructGenerator(const T &mergedGroup) : m_MergedGroup(mergedGroup)
    {
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addField(const std::string &type, const std::string &name, GetFieldValueFunc getFieldValue, FieldType fieldType = FieldType::Standard)
    {
        m_Fields.emplace_back(type, name, getFieldValue, fieldType);
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

    void addEGPs(const std::vector<Snippet::Base::EGP> &egps)
    {
        for(const auto &e : egps) {
            addField(e.type, e.name,
                     [e](const typename T::GroupInternal &g, size_t){ return e.name + g.getName(); },
                     Utils::isTypePointer(e.type) ? FieldType::PointerEGP : FieldType::ScalarEGP);
        }
    }

    template<typename G, typename H>
    void addHeterogeneousParams(const Snippet::Base::StringVec &paramNames, G getParamValues, H isHomogeneous)
    {
        // Loop through params
        for(size_t p = 0; p < paramNames.size(); p++) {
            // If parameters isn't homogeneous
            if(!(getMergedGroup().*isHomogeneous)(p)) {
                // Add field
                addField("scalar", paramNames[p],
                         [p, getParamValues](const typename T::GroupInternal &g, size_t)
                         {
                             const auto &values = (g.*getParamValues)();
                             return CodeGenerator::writePreciseString(values.at(p));
                         });
            }
        }
    }

    template<typename G, typename H>
    void addHeterogeneousDerivedParams(const Snippet::Base::DerivedParamVec &derivedParams, G getDerivedParamValues, H isHomogeneous)
    { 
        // Loop through derived params
        for(size_t p = 0; p < derivedParams.size(); p++) {
            // If parameters isn't homogeneous
            if(!(getMergedGroup().*isHomogeneous)(p)) {
                // Add field
                addField("scalar", derivedParams[p].name,
                         [p, getDerivedParamValues](const typename T::GroupInternal &g, size_t)
                         {
                             const auto &values = (g.*getDerivedParamValues)();
                             return CodeGenerator::writePreciseString(values.at(p));
                         });
            }
        }
    }

    template<typename V, typename H>
    void addHeterogeneousVarInitParams(const Models::Base::VarVec &vars, V getVarInitialisers, H isHomogeneous)
    {
        // Loop through weight update model variables
        const std::vector<Models::VarInit> &archetypeVarInitialisers = (getMergedGroup().getArchetype().*getVarInitialisers)();
        for(size_t v = 0; v < archetypeVarInitialisers.size(); v++) {
            // Loop through parameters
            const Models::VarInit &varInit = archetypeVarInitialisers[v];
            for(size_t p = 0; p < varInit.getParams().size(); p++) {
                if(!(getMergedGroup().*isHomogeneous)(v, p)) {

                    addField("scalar", varInit.getSnippet()->getParamNames()[p] + vars[v].name,
                             [p, v, getVarInitialisers](const typename T::GroupInternal &ng, size_t)
                             {
                                 const auto &values = (ng.*getVarInitialisers)()[v].getParams();
                                 return CodeGenerator::writePreciseString(values.at(p));
                             });
                }
            }
        }
    }

    template<typename V, typename H>
    void addHeterogeneousVarInitDerivedParams(const Models::Base::VarVec &vars, V getVarInitialisers, H isHomogeneous)
    {
        // Loop through weight update model variables
        const std::vector<Models::VarInit> &archetypeVarInitialisers = (getMergedGroup().getArchetype().*getVarInitialisers)();
        for(size_t v = 0; v < archetypeVarInitialisers.size(); v++) {
            // Loop through parameters
            const Models::VarInit &varInit = archetypeVarInitialisers[v];
            for(size_t d = 0; d < varInit.getDerivedParams().size(); d++) {
                if(!(getMergedGroup().*isHomogeneous)(v, d)) {
                    addField("scalar", varInit.getSnippet()->getDerivedParams()[d].name + vars[v].name,
                             [d, v, getVarInitialisers](const typename T::GroupInternal &ng, size_t)
                             {
                                 const auto &values = (ng.*getVarInitialisers)()[v].getDerivedParams();
                                 return CodeGenerator::writePreciseString(values.at(d));
                             });
                }
            }
        }
    }

    void generate(CodeGenerator::CodeStream &definitionsInternal, CodeGenerator::CodeStream &definitionsInternalFunc,
                  CodeGenerator::CodeStream &runnerVarDecl, CodeGenerator::CodeStream &runnerVarAlloc,
                  CodeGenerator::MergedEGPMap &mergedEGPs, const std::string &name)
    {
        const size_t index = getMergedGroup().getIndex();

        // Write struct declation to top of definitions internal
        definitionsInternal << "struct Merged" << name << "Group" << index << std::endl;
        {
            CodeGenerator::CodeStream::Scope b(definitionsInternal);
            for(const auto &f : m_Fields) {
                definitionsInternal << std::get<0>(f) << " " << std::get<1>(f) << ";" << std::endl;
            }
            definitionsInternal << std::endl;
        }

        definitionsInternal << ";" << std::endl;

        // Declare array of these structs containing individual neuron group pointers etc
        runnerVarDecl << "Merged" << name << "Group" << index << " merged" << name << "Group" << index << "[" << getMergedGroup().getGroups().size() << "];" << std::endl;

        for(size_t i = 0; i < getMergedGroup().getGroups().size(); i++) {
            const auto &g = getMergedGroup().getGroups()[i];

            // Set all fields in array of structs
            for(const auto &f : m_Fields) {
                const std::string fieldInitVal = std::get<2>(f)(g, i);
                runnerVarAlloc << "merged" << name << "Group" << index << "[" << i << "]." << std::get<1>(f) << " = " << fieldInitVal << ";" << std::endl;
                // If field is an EGP, add record to merged EGPS
                if(std::get<3>(f) != FieldType::Standard) {
                    mergedEGPs[fieldInitVal].emplace(
                        std::piecewise_construct, std::forward_as_tuple(name),
                        std::forward_as_tuple(index, i, (std::get<3>(f) == FieldType::PointerEGP), std::get<1>(f)));
                }
            }

        }

        // Then generate call to function to copy local array to device
        runnerVarAlloc << "pushMerged" << name << "Group" << index << "ToDevice(merged" << name << "Group" << index << ");" << std::endl;

        // Finally add declaration to function to definitions internal
        definitionsInternalFunc << "EXPORT_FUNC void pushMerged" << name << "Group" << index << "ToDevice(const Merged" << name << "Group" << index << " *group);" << std::endl;
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const T &getMergedGroup() const{ return m_MergedGroup; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const T &m_MergedGroup;
    std::vector<std::tuple<std::string, std::string, GetFieldValueFunc, FieldType>> m_Fields;
};

//--------------------------------------------------------------------------
// CodeGenerator::MergedNeuronStructGenerator
//--------------------------------------------------------------------------
class MergedNeuronStructGenerator : public MergedStructGenerator<CodeGenerator::NeuronGroupMerged>
{
public:
    MergedNeuronStructGenerator(const CodeGenerator::NeuronGroupMerged &mergedGroup)
    :   MergedStructGenerator<CodeGenerator::NeuronGroupMerged>(mergedGroup)
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
    MergedSynapseStructGenerator(const CodeGenerator::SynapseGroupMerged &mergedGroup)
    :   MergedStructGenerator<CodeGenerator::SynapseGroupMerged>(mergedGroup)
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
