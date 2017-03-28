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

    //------------------------------------------------------------------------
    // NeuronModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getSimCode() const{ return m_SimCode; }
    virtual std::string getThresholdConditionCode() const{ return m_ThresholdConditionCode; }
    virtual NewModels::Base::StringVec getParamNames() const{ return m_ParamNames; }
    virtual NewModels::Base::StringPairVec getVars() const{ return m_Vars; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_SimCode;
    std::string m_ThresholdConditionCode;
    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
};
}   // namespace SpineMLGenerator