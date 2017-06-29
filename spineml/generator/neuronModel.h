#pragma once

// Standard includes
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

//----------------------------------------------------------------------------
// SpineMLGenerator::NeuronModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class NeuronModel : public NeuronModels::Base
{
public:
    NeuronModel(const ModelParams::Neuron &params);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues VarValues;

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

    // How are send ports mapped to GeNN?
    std::set<std::string> m_SendPortVariables;
    std::string m_SendPortSpike;
    std::string m_SendPortSpikeLikeEvent;

    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
};
}   // namespace SpineMLGenerator