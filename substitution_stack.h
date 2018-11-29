#pragma once

// Standard C++ includes
#include <map>
#include <stdexcept>
#include <string>

// Standard C includes
#include <cassert>

// GeNN includes
#include "codeGenUtils.h"

class Substitutions
{
public:
    Substitutions(const Substitutions *parent = nullptr) : m_Parent(parent)
    {
        assert(m_Parent != this);
    }

    void addVarSubstitution(const std::string &source, const std::string &destionation) 
    {
        m_VarSubstitutions.emplace(source, destionation);
    }

    void addFuncSubstitution(const std::string &source, unsigned int numArguments, const std::string &funcTemplate)
    {
        m_FuncSubstitutions.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(source),
                                    std::forward_as_tuple(numArguments, funcTemplate));
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
        // Apply variable substitutions
        for(const auto &v : m_VarSubstitutions) {
            substitute(code, "$(" + v.first + ")", v.second);
        }

        // Apply function substitutions
        for(const auto &f : m_FuncSubstitutions) {
            functionSubstitute(code, f.first, f.second.first, f.second.second);
        }

        // If we have a parent, apply their substitutions too
        if(m_Parent) {
            m_Parent->apply(code);
        }
    }

private:
    std::map<std::string, std::string> m_VarSubstitutions;
    std::map<std::string, std::pair<unsigned int, std::string>> m_FuncSubstitutions;
    const Substitutions *m_Parent;
};
