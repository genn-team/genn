#pragma once

// Standard C++ includes
#include <unordered_map>

//----------------------------------------------------------------------------
// ParamValues
//----------------------------------------------------------------------------
//! Container for parameter (and derived parameter) values
/*! We don't use std::unordered_map directly both to save typing and because it doesn't have a const [] operator */
class ParamValues
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::unordered_map<std::string, double> ParamMap;

    ParamValues()
    {
    }

    ParamValues(const ParamMap &values)
    :   m_Values(values)
    {}

    ParamValues(std::initializer_list<ParamMap::value_type> values)
    :   m_Values(values)
    {}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    double operator[](const std::string &name) const
    {
        return m_Values.at(name);
    }

    ParamMap &getValues() { return m_Values; }
    const ParamMap &getValues() const { return m_Values; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    ParamMap m_Values;
};