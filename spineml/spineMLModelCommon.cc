#include "spineMLModelCommon.h"

// Standard C++ includes
#include <algorithm>

// GeNN includes
#include "newModels.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::ParamValues
//----------------------------------------------------------------------------
std::vector<double> SpineMLGenerator::ParamValues::getValues() const
{
    // Get parameter names from model
    auto modelParamNames = m_Model.getParamNames();

    // Reserve vector of values to match it
    std::vector<double> paramValues;
    paramValues.reserve(modelParamNames.size());

    // Populate this vector with either values from map or 0s
    std::transform(modelParamNames.begin(), modelParamNames.end(),
                   std::back_inserter(paramValues),
                   [this](const std::string &n)
                   {
                       auto value = m_Values.find(n);
                       if(value == m_Values.end()) {
                           return 0.0;
                       }
                       else {
                           return value->second;
                       }
                   });
    return paramValues;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::VarValues
//----------------------------------------------------------------------------
std::vector<double> SpineMLGenerator::VarValues::getValues() const
{
    // Get variables from model
    auto modelVars = m_Model.getVars();

    // Reserve vector of values to match it
    std::vector<double> varValues;
    varValues.reserve(modelVars.size());

    // Populate this vector with either values from map or 0s
    std::transform(modelVars.begin(), modelVars.end(),
                   std::back_inserter(varValues),
                   [this](const std::pair<std::string, std::string> &n)
                   {
                       auto value = m_Values.find(n.first);
                       if(value == m_Values.end()) {
                           return 0.0;
                       }
                       else {
                           return value->second;
                       }
                   });
    return varValues;
}