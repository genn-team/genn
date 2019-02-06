#pragma once

// Standard includes
#include <algorithm>
#include <map>
#include <set>
#include <string>

// GeNN includes
#include "newNeuronModels.h"

// Spine ML generator includes
#include "modelCommon.h"

// Forward declarations
namespace SpineMLGenerator
{
    namespace ModelParams
    {
        class Neuron;
    }
}

namespace pugi
{
    class xml_node;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::NeuronModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class NeuronModel : public NeuronModels::Base
{
public:
    NeuronModel(const ModelParams::Neuron &params, const pugi::xml_node &componentClass);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues<NeuronModel> VarValues;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    const std::string &getSendPortSpike() const
    {
        return m_SendPortSpike;
    }

    const std::string &getSendPortSpikeLikeEvent() const
    {
        return m_SendPortSpikeLikeEvent;
    }

    const std::set<std::string> &getSendPortVariables() const
    {
        return m_SendPortVariables;
    }

    bool hasSendPortVariable(const std::string &port) const
    {
        return (m_SendPortVariables.find(port) != m_SendPortVariables.end());
    }

    bool hasAdditionalInputVar(const std::string &port) const
    {
        auto iVar = std::find_if(m_AdditionalInputVars.begin(), m_AdditionalInputVars.end(),
                                 [port](const std::pair<std::string, std::pair<std::string, double>> &var)
                                 {
                                     return (var.first == port);
                                 });
        return (iVar != m_AdditionalInputVars.end());
    }

    unsigned int getInitialRegimeID() const
    {
        return m_InitialRegimeID;
    }

    //------------------------------------------------------------------------
    // NeuronModels::Base virtuals
    //------------------------------------------------------------------------
    virtual std::string getSimCode() const{ return m_SimCode; }
    virtual std::string getThresholdConditionCode() const{ return m_ThresholdConditionCode; }

    virtual Models::Base::StringVec getParamNames() const{ return m_ParamNames; }
    virtual Models::Base::StringPairVec getVars() const{ return m_Vars; }
    virtual Models::Base::NameTypeValVec getAdditionalInputVars() const{ return m_AdditionalInputVars; }
    virtual Models::Base::DerivedParamVec getDerivedParams() const{ return m_DerivedParams; }

    //------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------
    static const char *componentClassName;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_SimCode;
    std::string m_ThresholdConditionCode;

    // How are send ports mapped to GeNN?
    std::set<std::string> m_SendPortVariables;
    std::string m_SendPortSpike;
    std::string m_SendPortSpikeLikeEvent;

    // GeNN model data
    Models::Base::NameTypeValVec m_AdditionalInputVars;
    Models::Base::StringVec m_ParamNames;
    Models::Base::StringPairVec m_Vars;
    Models::Base::DerivedParamVec m_DerivedParams;

    unsigned int m_InitialRegimeID;
};
}   // namespace SpineMLGenerator