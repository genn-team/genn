#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>
#include <vector>

// GeNN includes
#include "newNeuronModels.h"

// Forward declarations
namespace pugi
{
    class xml_node;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLNeuronModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class SpineMLNeuronModel : public NeuronModels::Base
{
public:
    SpineMLNeuronModel(const pugi::xml_node &neuronNode,
                       const std::string &url, const std::set<std::string> &variableParams);

    //------------------------------------------------------------------------
    // ParamValues
    //------------------------------------------------------------------------
    class ParamValues
    {
    public:
        ParamValues(const std::map<std::string, double> &values) : m_Values(values){}

        //----------------------------------------------------------------------------
        // Public API
        //----------------------------------------------------------------------------
        std::vector<double> getValues() const;

    private:
        //----------------------------------------------------------------------------
        // Members
        //----------------------------------------------------------------------------
        const std::map<std::string, double> &m_Values;
    };
    typedef ParamValues VarValues;
};
}   // namespace SpineMLGenerator