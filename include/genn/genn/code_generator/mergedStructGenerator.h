#pragma once

// Standard C++ includes
#include <functional>
#include <vector>

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
    // Typedefines
    //------------------------------------------------------------------------
    typedef std::function<std::string(const typename T::GroupInternal &, size_t)> GetFieldValueFunc;

    MergedStructGenerator(const T &mergedGroup) : m_MergedGroup(mergedGroup)
    {
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addField(const std::string &name, GetFieldValueFunc getFieldValue)
    {
        m_Fields.emplace_back(name, getFieldValue);
    }

    void addPointerField(const std::string &name, const std::string &prefix)
    {
        addField(name, [prefix](const typename T::GroupInternal &g, size_t){ return prefix + g.getName(); });
    }

    void addVars(const std::vector<Models::Base::Var> &vars, const std::string &prefix)
    {
        for(const auto &v : vars) {
            addPointerField(v.type + " *" + v.name, prefix + v.name);
        }
    }

    void addEGPs(const std::vector<Snippet::Base::EGP> &egps)
    {
        for(const auto &e : egps) {
            addField(e.type + " " + e.name,
                     [e](const typename T::GroupInternal &g, size_t){ return e.name + g.getName(); });
        }
    }

    void generate(CodeGenerator::CodeStream &definitionsInternal, CodeGenerator::CodeStream &definitionsInternalFunc,
                  CodeGenerator::CodeStream &runnerVarAlloc, const std::string &name)
    {
        const size_t index = getMergedGroup().getIndex();

        // Write struct declation to top of definitions internal
        definitionsInternal << "struct Merged" << name << "Group" << index << std::endl;
        {
            CodeGenerator::CodeStream::Scope b(definitionsInternal);
            for(const auto &f : m_Fields) {
                definitionsInternal << f.first << ";" << std::endl;
            }
            definitionsInternal << std::endl;
        }

        definitionsInternal << ";" << std::endl;

        // Write local array of these structs containing individual neuron group pointers etc
        // **NOTE** scope will hopefully reduce stack usage
        {
            CodeStream::Scope b(runnerVarAlloc);
            runnerVarAlloc << "Merged" << name << "Group" << index << " merged" << name << "Group" << index << "[] = ";
            {
                CodeGenerator::CodeStream::Scope b(runnerVarAlloc);
                for(size_t i = 0; i < getMergedGroup().getGroups().size(); i++) {
                    const auto &sg = getMergedGroup().getGroups()[i];

                    runnerVarAlloc << "{";
                    for(const auto &f : m_Fields) {
                        runnerVarAlloc << f.second(sg, i) << ", ";
                    }
                    runnerVarAlloc << "}," << std::endl;
                }
            }
            runnerVarAlloc << ";" << std::endl;

            // Then generate call to function to copy local array to device
            runnerVarAlloc << "pushMerged" << name << "Group" << index << "ToDevice(merged" << name << "Group" << index << ");" << std::endl;
        }
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
    std::vector<std::pair<std::string, GetFieldValueFunc>> m_Fields;
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
    void addMergedInSynPointerField(const std::string &name, size_t archetypeIndex, const std::string &prefix,
                                    const std::vector<std::vector<std::pair<SynapseGroupInternal*, std::vector<SynapseGroupInternal*>>>> &sortedMergedInSyns)
    {
        addField(name + std::to_string(archetypeIndex),
                 [prefix, &sortedMergedInSyns, archetypeIndex](const NeuronGroupInternal&, size_t groupIndex)
                 {
                     return prefix + sortedMergedInSyns.at(groupIndex).at(archetypeIndex).first->getPSModelTargetName();
                 });
    }

    void addCurrentSourcePointerField(const std::string &name, size_t archetypeIndex, const std::string &prefix,
                                      const std::vector<std::vector<CurrentSourceInternal*>> &sortedCurrentSources)
    {
        addField(name + std::to_string(archetypeIndex),
                 [prefix, &sortedCurrentSources, archetypeIndex](const NeuronGroupInternal&, size_t groupIndex)
                 {
                     return prefix + sortedCurrentSources.at(groupIndex).at(archetypeIndex)->getName();
                 });
    }

    void addSynPointerField(const std::string &name, size_t archetypeIndex, const std::string &prefix,
                            const std::vector<std::vector<SynapseGroupInternal*>> &sortedSyn)
    {
        addField(name + std::to_string(archetypeIndex),
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
    void addPSPointerField(const std::string &name, const std::string &prefix)
    {
        addField(name, [prefix](const SynapseGroupInternal &sg, size_t){ return prefix + sg.getPSModelTargetName(); });
    }

    void addSrcPointerField(const std::string &name, const std::string &prefix)
    {
        addField(name, [prefix](const SynapseGroupInternal &sg, size_t){ return prefix + sg.getSrcNeuronGroup()->getName(); });
    }

    void addTrgPointerField(const std::string &name, const std::string &prefix)
    {
        addField(name, [prefix](const SynapseGroupInternal &sg, size_t){ return prefix + sg.getTrgNeuronGroup()->getName(); });
    }
};

}   // namespace CodeGenerator
