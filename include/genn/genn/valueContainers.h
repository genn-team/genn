#pragma once

// Standard C++ includes
#include <unordered_map>

//----------------------------------------------------------------------------
// ValueContainerBase
//----------------------------------------------------------------------------
//! Base class for containers for named values passed by user
/*! We don't use std::unordered_map directly both to save typing and because it doesn't have a const [] operator */
template<typename V>
class ValueContainerBase
{
public:
    //----------------------------------------------------------------------------
    // Typedefines
    //----------------------------------------------------------------------------
    typedef std::unordered_map<std::string, V> MapType;

    ValueContainerBase()
    {
    }

    ValueContainerBase(const MapType &values)
    :   m_Values(values)
    {}

    ValueContainerBase(std::initializer_list<typename MapType::value_type> values)
    :   m_Values(values)
    {}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    V operator[](const std::string &name) const
    {
        return m_Values.at(name);
    }

    MapType &getValues() { return m_Values; }
    const MapType &getValues() const { return m_Values; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    MapType m_Values;
};

//----------------------------------------------------------------------------
// ParamValues
//----------------------------------------------------------------------------
//! Container for parameter (and derived parameter) values
using ParamValues = ValueContainerBase<double>;
