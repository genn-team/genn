#pragma once

// Standard C++ includes
#include <map>
#include <stdexcept>
#include <string>

// Standard C includes
#include <cassert>

// GeNN code generator includes
#include "codeGenUtils.h"

//--------------------------------------------------------------------------
// Substitutions
//--------------------------------------------------------------------------
namespace CodeGenerator
{
class Substitutions
{
public:
    Substitutions(const Substitutions *parent = nullptr) : m_Parent(parent)
    {
        assert(m_Parent != this);
    }

    Substitutions(const std::vector<FunctionTemplate> &functions) : m_Parent(nullptr)
    {
        // Loop through functions and add as substitutions
        for(const auto &f: functions) {
            addFuncSubstitution(f.genericName, f.numArguments, f.functionTemplate);
        }
    }

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    void addVarSubstitution(const std::string &source, const std::string &destionation, bool allowOverride = false)
    {
        auto res = m_VarSubstitutions.emplace(source, destionation);
        if(!allowOverride && !res.second) {
            throw std::runtime_error("'" + source + "' already has a variable substitution");
        }
    }

    void addFuncSubstitution(const std::string &source, unsigned int numArguments, const std::string &funcTemplate, bool allowOverride = false)
    {
        auto res = m_FuncSubstitutions.emplace(std::piecewise_construct,
                                               std::forward_as_tuple(source),
                                               std::forward_as_tuple(numArguments, funcTemplate));
        if(!allowOverride && !res.second) {
            throw std::runtime_error("'" + source + "' already has a function substitution");
        }
    }

    bool hasVarSubstitution(const std::string &source) const
    {
        return (m_VarSubstitutions.find(source) != m_VarSubstitutions.end());
    }

    const std::string &getVarSubstitution(const std::string &source) const
    {
        auto var = m_VarSubstitutions.find(source);
        if(var != m_VarSubstitutions.end()) {
            return var->second;
        }
        else if(m_Parent) {
            return m_Parent->getVarSubstitution(source);
        }
        else {
            throw std::runtime_error("Nothing to substitute for '" + source + "'");
        }
    }

    void apply(std::string &code) const
    {
        // Apply function and variable substitutions
        // **NOTE** functions may contain variables so evaluate ALL functions first
        applyFuncs(code);
        applyVars(code);
    }

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    const std::string operator[] (const std::string &source) const
    {
        return getVarSubstitution(source);
    }

private:
    //--------------------------------------------------------------------------
    // Private API
    //--------------------------------------------------------------------------
    void applyFuncs(std::string &code) const
    {
        // Apply function substitutions
        for(const auto &f : m_FuncSubstitutions) {
            functionSubstitute(code, f.first, f.second.first, f.second.second);
        }

        // If we have a parent, apply their function substitutions too
        if(m_Parent) {
            m_Parent->applyFuncs(code);
        }
    }

    void applyVars(std::string &code) const
    {
        // Apply variable substitutions
        for(const auto &v : m_VarSubstitutions) {
            substitute(code, "$(" + v.first + ")", v.second);
        }

        // If we have a parent, apply their variable substitutions too
        if(m_Parent) {
            m_Parent->applyVars(code);
        }
    }

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    std::map<std::string, std::string> m_VarSubstitutions;
    std::map<std::string, std::pair<unsigned int, std::string>> m_FuncSubstitutions;
    const Substitutions *m_Parent;
};
}   // namespace CodeGenerator
