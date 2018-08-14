#pragma once

// Standard C++ includes
#include <map>
#include <stdexcept>
#include <string>

// Standard C includes
#include <cassert>

// GeNN includes
#include "codeGenUtils.h"

//**TODO** this is crap - replace by chain of std::maps
class Substitutions
{
public:
    Substitutions(const Substitutions *parent = nullptr) : m_Parent(parent)
    {
    }

    void addSubstitution(const std::string &source, const std::string &destionation) 
    {
        m_Substitutions.emplace(source, destionation);
    }

    const std::string &getSubstitution(const std::string &source) const
    {
        auto var = m_Substitutions.find(source);
        if(var != m_Substitutions.end()) {
            return var->second;
        }
        else if(m_Parent) {
            return m_Parent->getSubstitution(source);
        }
        else {
            throw std::runtime_error("Nothing to substitute for '" + source + "'");
        }
    }

    void apply(std::string &code) const
    {
        // Apply substitutions
        for(const auto &s : m_Substitutions) {
            substitute(code, "$(" + s.first + ")", s.second);
        }

        // If we have a parent, apply their substitutions too
        if(m_Parent) {
            m_Parent->apply(code);
        }
    }

private:
    std::map<std::string, std::string> m_Substitutions;
    const Substitutions *m_Parent;
};