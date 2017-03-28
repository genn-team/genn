#pragma once

// Standard includes
#include <map>
#include <string>
#include <vector>

// Forward declarations
namespace NewModels
{
    class Base;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ParamValues
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class ParamValues
{
public:
    ParamValues(const std::map<std::string, double> &values, const NewModels::Base &model)
        : m_Values(values), m_Model(model){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::vector<double> getValues() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const std::map<std::string, double> &m_Values;
    const NewModels::Base &m_Model;
};

//------------------------------------------------------------------------
// VarValues
//------------------------------------------------------------------------
class VarValues
{
public:
    VarValues(const std::map<std::string, double> &values, const NewModels::Base &model)
        : m_Values(values), m_Model(model){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::vector<double> getValues() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const std::map<std::string, double> &m_Values;
    const NewModels::Base &m_Model;
};
}   // namespace SpineMLGenerator