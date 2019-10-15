#include "postsynapticModel.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <regex>

// Standard C includes
#include <cstring>

// pugixml includes
#include "pugixml/pugixml.hpp"

// PLOG includes
#include <plog/Log.h>

// SpineML common includes
#include "spineMLUtils.h"

// Spine ML generator includes
#include "modelParams.h"
#include "objectHandler.h"
#include "neuronModel.h"
#include "weightUpdateModel.h"

using namespace SpineMLCommon;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// ObjectHandlerImpulse
//----------------------------------------------------------------------------
class ObjectHandlerImpulse : public SpineMLGenerator::ObjectHandler::Base
{
public:
    ObjectHandlerImpulse(const std::string &spikeImpulseReceivePort)
        : m_SpikeImpulseReceivePort(spikeImpulseReceivePort)
    {

    }

    //----------------------------------------------------------------------------
    // SpineMLGenerator::ObjectHandler::Base virtuals
    //----------------------------------------------------------------------------
    void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                  unsigned int targetRegimeID)
    {
        if(currentRegimeID != targetRegimeID) {
            throw std::runtime_error("GeNN cannot handle postsynaptic models where impulses cause a regime-change");
        }

        auto stateAssign = node.child("StateAssignment");
        if(!stateAssign) {
            throw std::runtime_error("GeNN only supports postsynaptic models where state assignment occurs");
        }

        // Get name of the variable state is assigned to
        m_ImpulseAssignStateVar = stateAssign.attribute("variable").value();

        // Match for A + B type expression with any amount of whitespace
        auto stateAssigmentCode = stateAssign.child_value("MathInline");
        std::regex regex("\\s*([a-zA-Z_]+)\\s*\\+\\s*([a-zA-Z_]+)\\s*");
        std::cmatch match;
        if(std::regex_match(stateAssigmentCode, match, regex)) {
            // If match is successful check that the two variables being added are the state variable and the spike impulse
            if((match[1].str() == m_SpikeImpulseReceivePort && match[2].str() == m_ImpulseAssignStateVar)
                || (match[1].str() == m_ImpulseAssignStateVar && match[2].str() == m_SpikeImpulseReceivePort))
            {
                return;
            }
        }
        throw std::runtime_error("GeNN only supports postsynaptic models which add incoming weight to state variable");
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const std::string &getImpulseAssignStateVar() const{ return m_ImpulseAssignStateVar; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::string m_SpikeImpulseReceivePort;
    std::string m_ImpulseAssignStateVar;
};

//----------------------------------------------------------------------------
// ObjectHandlerEvent
//----------------------------------------------------------------------------
class ObjectHandlerEvent : public SpineMLGenerator::ObjectHandler::Base
{
public:
    //----------------------------------------------------------------------------
    // SpineMLGenerator::ObjectHandler::Base virtuals
    //----------------------------------------------------------------------------
    void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                  unsigned int targetRegimeID)
    {
        if(currentRegimeID != targetRegimeID) {
            throw std::runtime_error("GeNN cannot handle postsynaptic models where events cause a regime-change");
        }

        // If this event doesn't output a a single event
        auto outgoingEvents = node.children("EventOut");
        const size_t numOutgoingEvents = std::distance(outgoingEvents.begin(), outgoingEvents.end());
        if(numOutgoingEvents != 1) {
            throw std::runtime_error("GeNN only supports postsynaptic models which handle events by emitting a single event");
        }
    }
};
}   // Anonymous namespace


//----------------------------------------------------------------------------
// SpineMLGenerator::PostsynapticModel
//----------------------------------------------------------------------------
const char *SpineMLGenerator::PostsynapticModel::componentClassName = "postsynapse";
//----------------------------------------------------------------------------
SpineMLGenerator::PostsynapticModel::PostsynapticModel(const ModelParams::Postsynaptic &params,
                                                       const pugi::xml_node &componentClass,
                                                       const NeuronModel *trgNeuronModel,
                                                       const WeightUpdateModel *weightUpdateModel)
{
    // Loop through send ports
    LOGD << "\t\tSend ports:";
    std::vector<std::tuple<std::string, std::string, bool>> sendPortVariables;
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();

        // If this is an analogue send port or event send port
        if(nodeType == "AnalogSendPort" || nodeType == "EventSendPort") {
            // Find the name of the port on the neuron which this send port targets
            const auto &neuronPortTrg = params.getOutputPortTrg(portName);
            if(neuronPortTrg.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON) {
                if(trgNeuronModel->hasAdditionalInputVar(neuronPortTrg.second)) {
                    LOGD << "\t\t\tImplementing " << nodeType << " '" << portName << "' using postsynaptic neuron additional input var '" << neuronPortTrg.second << "'";

                    // Add mapping to vector
                    sendPortVariables.push_back(std::make_tuple(neuronPortTrg.second, portName, nodeType == "EventSendPort"));
                }
                else {
                    throw std::runtime_error("Post synaptic models can only provide input to impulse receive ports");
                }
            }
            else {
                throw std::runtime_error("GeNN does not support " + nodeType + " which target anything other than postsynaptic neurons");
            }
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' send ports in postsynaptic models");
        }
    }

    // Possible means by which this postsynaptic model can receive input from the weight update model
    enum class WUMInputType
    {
        None,
        AnalogueReduce,
        ImpulseReceive,
        EventReceive,
    };

    // Read aliases
    std::map<std::string, std::string> aliases;
    readAliases(componentClass, aliases);

    // Loop through receive ports
    LOGD << "\t\tReceive ports:";
    std::map<std::string, std::string> receivePortVariableMap;
    PortTypeName<WUMInputType, WUMInputType::None> wumInputPort;
    for(auto receivePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReceivePort").c_str())) {
        std::string nodeType = receivePort.node().name();
        const char *portName = receivePort.node().attribute("name").value();
        const auto &portSrc = params.getInputPortSrc(portName);

        // If this port is an analogue receive port for some sort of postsynaptic neuron state variable
        if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON
            && trgNeuronModel->hasSendPortVariable(portSrc.second))
        {
            LOGD << "\t\t\tImplementing analogue receive port '" << portName << "' using postsynaptic neuron send port variable '" << portSrc.second << "'";
            receivePortVariableMap.emplace(portName, portSrc.second);
        }
        // Otherwise if this port is an impulse receive port which receives spike impulses from weight update model
        else if(nodeType == "ImpulseReceivePort" && portSrc.first == ModelParams::Base::PortSource::WEIGHT_UPDATE
            && weightUpdateModel->getSendPortSpikeImpulse() == portSrc.second)
        {
            LOGD << "\t\t\tImplementing impulse receive port '" << portName << "' as GeNN weight update model input";
            wumInputPort.set(WUMInputType::ImpulseReceive, portName);
        }
        // Otherwise if this port is an event receive port which receives spikes from weight update model
        else if(nodeType == "EventReceivePort" && portSrc.first == ModelParams::Base::PortSource::WEIGHT_UPDATE
            && weightUpdateModel->getSendPortSpikeImpulse() == portSrc.second)
        {
            LOGD << "\t\t\tImplementing event receive port '" << portName << "' as GeNN weight update model input";
            wumInputPort.set(WUMInputType::EventReceive, portName);
        }
        else
        {
            throw std::runtime_error("GeNN does not currently support '" + nodeType + "' receive ports in postsynaptic models");
        }
    }

    // Loop through reduce ports
    LOGD << "\t\tReduce ports:";
    for(auto reducePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReducePort").c_str())) {
        std::string nodeType = reducePort.node().name();
        const char *portName = reducePort.node().attribute("name").value();
        const auto &portSrc = params.getInputPortSrc(portName);

        // If this is an analogue reduce port which receives analogue input from weight update model
        if(nodeType == "AnalogReducePort" && portSrc.first == ModelParams::Base::PortSource::WEIGHT_UPDATE
            && weightUpdateModel->getSendPortAnalogue() == portSrc.second)
        {
            LOGD << "\t\t\tImplementing analogue reduce port '" << portName << "' as GeNN weight update model input";
            wumInputPort.set(WUMInputType::AnalogueReduce, portName);
        }
        else
        {
            throw std::runtime_error("GeNN does not currently support '" + nodeType + "' reduce ports in postsynaptic models");
        }
    }

    // Create a code stream for generating decay code
    CodeStream decayCodeStream;

    // Create lambda function to end regime on all code streams when required
    auto regimeEndFunc =
        [&decayCodeStream]
        (bool multipleRegimes, unsigned int currentRegimeID)
        {
            decayCodeStream.onRegimeEnd(multipleRegimes, currentRegimeID);
        };

    // Generate model code using specified condition handler
    bool multipleRegimes;
    ObjectHandler::Condition objectHandlerCondition(decayCodeStream, aliases);
    ObjectHandlerImpulse objectHandlerImpulse(wumInputPort.getName(WUMInputType::ImpulseReceive));
    ObjectHandlerEvent objectHandlerEvent;
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(decayCodeStream, aliases);
    std::tie(multipleRegimes, m_InitialRegimeID) = generateModelCode(componentClass,
                                                                     {
                                                                         { wumInputPort.getName(WUMInputType::EventReceive), &objectHandlerEvent }
                                                                     },
                                                                     &objectHandlerCondition,
                                                                     {
                                                                         { wumInputPort.getName(WUMInputType::ImpulseReceive), &objectHandlerImpulse }
                                                                     },
                                                                     &objectHandlerTimeDerivative, regimeEndFunc);

    // Store generated code in class
    m_DecayCode = decayCodeStream.str();

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, params.getVariableParams(), multipleRegimes);

    // Add any derived parameters required for time-derivative
    objectHandlerTimeDerivative.addDerivedParams(m_ParamNames, m_DerivedParams);

    // Determine what state variable inpulse is applied to
    const std::string &impulseAssignStateVar = objectHandlerImpulse.getImpulseAssignStateVar();

    // Loop through send port variables
    for(const auto s : sendPortVariables) {
        // If this is an event send port
        if(std::get<2>(s)) {
            m_ApplyInputCode += std::get<0>(s) + " += $(inSyn); $(inSyn) = 0;\n";
        }
        // Otherwise, if it's an analogue send port
        else {
            // Resolve any aliases to get value to send through this send port
            std::string inputCode = getSendPortCode(aliases, m_Vars, std::get<1>(s));

            // If input to postsynaptic model is provided as an impulse, use the
            // internal $(inSyn) variable in place of the state variable input is added to
            if(wumInputPort.getType() == WUMInputType::ImpulseReceive) {
                wrapAndReplaceVariableNames(inputCode, impulseAssignStateVar, "inSyn");
            }
            // Otherwise, if input is provided through an analogue reduce port, use the
            // internal $(inSyn) variable in place of the name oft he reduce port
            else if(wumInputPort.getType() == WUMInputType::AnalogueReduce) {
                wrapAndReplaceVariableNames(inputCode, wumInputPort.getName(), "inSyn");
            }

            // Generate code to write to send port
            m_ApplyInputCode += std::get<0>(s) + " += " + inputCode + ";";

            // If there is no decay code, zero the synaptic input after it's read
            if(m_DecayCode.empty()) {
                m_ApplyInputCode += "$(inSyn) = 0;";
            }

            // Add a line ending
            m_ApplyInputCode += "\n";
        }
    }

    // If incoming impulse is being assigned to a state variable
    if(wumInputPort.getType() == WUMInputType::ImpulseReceive && !impulseAssignStateVar.empty()) {
        LOGD << "\t\tImpulse assign state variable:" << impulseAssignStateVar;

        // Substitute name of analogue send port for internal variable
        wrapAndReplaceVariableNames(m_DecayCode, impulseAssignStateVar, "inSyn");

        // As this variable is being implemented using a built in GeNN state variable, remove it from variables
        auto stateVar = std::find_if(m_Vars.begin(), m_Vars.end(),
                                     [impulseAssignStateVar](const Models::Base::Var &var)
                                     {
                                         return (var.name == impulseAssignStateVar);
                                     });
        if(stateVar == m_Vars.end()) {
            throw std::runtime_error("State variable '" + impulseAssignStateVar + "' referenced in OnImpulse not found");
        }
        else {
            m_Vars.erase(stateVar);
        }
    }

    // Correctly wrap and replace references to receive port variable in code string
    for(const auto &r : receivePortVariableMap) {
        wrapAndReplaceVariableNames(m_DecayCode, r.first, r.second);
        wrapAndReplaceVariableNames(m_ApplyInputCode, r.first, r.second);
    }

    // Correctly wrap references to parameters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars, m_DerivedParams,
                             {&m_DecayCode, &m_ApplyInputCode});
}
