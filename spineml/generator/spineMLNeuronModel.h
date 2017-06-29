#pragma once

// Standard includes
#include <map>
#include <set>
#include <string>

// GeNN includes
#include "newNeuronModels.h"

// Spine ML generator includes
#include "spineMLModelCommon.h"

// Forward declarations
namespace SpineMLGenerator
{
    namespace ModelParams
    {
        class Neuron;
    }
}

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLNeuronModel
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class SpineMLNeuronModel : public NeuronModels::Base
{
public:
    SpineMLNeuronModel(const ModelParams::Neuron &params);

    //------------------------------------------------------------------------
    // Typedefines
    //------------------------------------------------------------------------
    typedef SpineMLGenerator::ParamValues ParamValues;
    typedef SpineMLGenerator::VarValues VarValues;

    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    // Possible means by which neuron send ports can be mapped to GeNN
    enum class SendPort
    {
        VARIABLE,
        SPIKE,
        SPIKE_LIKE_EVENT,
    };

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
    std::map<std::string, SendPort> m_SendPortMappings;

    NewModels::Base::StringVec m_ParamNames;
    NewModels::Base::StringPairVec m_Vars;
};
}   // namespace SpineMLGenerator