#include "neuronModel.h"

// Standard C++ includes
#include <iostream>
#include <regex>
#include <sstream>

// Standard C includes
#include <cstring>

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML common includes
#include "spineMLLogging.h"
#include "spineMLUtils.h"

// Spine ML generator includes
#include "modelParams.h"
#include "objectHandler.h"

using namespace SpineMLCommon;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// RegimeThresholds
//----------------------------------------------------------------------------
class RegimeThresholds
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void addTriggerCode(unsigned int regimeID, const std::string &triggerCodeString) {
        // If there isn't already a piece of trigger code for this regime, add one
        // **THINK** is this a fair restriction - if not should they be ORed or ANDed together?
        if(!m_RegimeThresholds.emplace(regimeID, triggerCodeString).second) {
            throw std::runtime_error("Only one spike trigger is supported per regime");
        }
    }

    std::string getThresholdCode(bool multipleRegimes) const
    {
        // If there are multiple regimes
        std::ostringstream thresholdCodeStream;
        if(multipleRegimes) {
            // Loop through them
            for(const auto &r : m_RegimeThresholds) {
                // If there are existing threshold conditions, OR them with this one
                if(thresholdCodeStream.tellp() > 0) {
                    thresholdCodeStream << " || ";
                }

                // Add test, ANDing test for correct regime ID with threshold condition
                thresholdCodeStream << "(_regimeID == " << r.first << " && (" << r.second << "))";
            }
        }
        // Otherwise, if there are any threshold tests
        else if(!m_RegimeThresholds.empty()) {
            if(m_RegimeThresholds.size() > 1) {
                throw std::runtime_error("Multiple regimes have not been found but there are thresholds specified for different regimes");
            }

            // Code should simple be that test
            thresholdCodeStream << "(" << m_RegimeThresholds.begin()->second << ")";
        }
        return thresholdCodeStream.str();

    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::map<unsigned int, std::string> m_RegimeThresholds;
};

//----------------------------------------------------------------------------
// ObjectHandlerCondition
//----------------------------------------------------------------------------
class ObjectHandlerCondition : public SpineMLGenerator::ObjectHandler::Condition
{
public:
    ObjectHandlerCondition(SpineMLGenerator::CodeStream &codeStream, const std::string &sendPortSpike, RegimeThresholds &regimeThresholds)
        : SpineMLGenerator::ObjectHandler::Condition(codeStream), m_SendPortSpike(sendPortSpike), m_RegimeThresholds(regimeThresholds){}

    //----------------------------------------------------------------------------
    // SpineMLGenerator::ObjectHandler::Condition virtuals
    //----------------------------------------------------------------------------
    void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                  unsigned int targetRegimeID)
    {
        // Superclass
        SpineMLGenerator::ObjectHandler::Condition::onObject(node, currentRegimeID,
                                                             targetRegimeID);
        // If this condition emits a spike
        // **TODO** also handle spike like event clause
        pugi::xpath_variable_set spikeEventsOutVars;
        spikeEventsOutVars.set("portName", m_SendPortSpike.c_str());
        auto spikeEventOut = node.select_node("EventOut[@port=$portName]", &spikeEventsOutVars);
        if(spikeEventOut) {
            // Add current regime and trigger condition to map
            // **NOTE** cannot build code immediately as we don't know if there are multiple regimes
            m_RegimeThresholds.addTriggerCode(currentRegimeID, node.child("Trigger").child("MathInline").text().get());
        }
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::string m_SendPortSpike;
    RegimeThresholds &m_RegimeThresholds;
};

//----------------------------------------------------------------------------
// ObjectHandlerEvent
//----------------------------------------------------------------------------
class ObjectHandlerEvent : public SpineMLGenerator::ObjectHandler::Base
{
public:
    ObjectHandlerEvent(SpineMLGenerator::CodeStream &codeStream, RegimeThresholds &regimeThresholds)
        : m_CodeStream(codeStream), m_RegimeThresholds(regimeThresholds){}

    //------------------------------------------------------------------------
    // ObjectHandler::Base virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID)
    {
        // If event is handled with a state assignement
        const std::string srcPort = node.attribute("src_port").value();
        if(auto stateAssigment = node.child("StateAssignment")) {
            // Extract variable name
            const std::string variableName = stateAssigment.attribute("variable").value();

            // Match for A + B type expression with any amount of whitespace
            // **TODO** generalise this - anything linear on variable would be fine
            auto stateAssigmentCode = stateAssigment.child_value("MathInline");
            std::regex regex("\\s*([a-zA-Z_]+)\\s*\\+\\s*([a-zA-Z_]+)\\s*");
            std::cmatch match;
            if(std::regex_match(stateAssigmentCode, match, regex)) {
                // If match is successful, write state assignement code to add scaled source port value to variable name
                if(match[1].str() == variableName) {
                    writeStateAssignement(variableName, match[2].str(), srcPort);
                }
                else if(match[2].str() == variableName) {
                    writeStateAssignement(variableName, match[1].str(), srcPort);
                }
                else {
                    throw std::runtime_error("GeNN only supports neuron models which multiply event count by variable");
                }
            }
            else {
                throw std::runtime_error("GeNN only supports neuron models which perform linear computation on incoming events");
            }
        }
        // Otherwise if this event should simply trigger an output event, add a trigger
        // to the regime's threshold condition to test if any events have been reveived
        else if(auto eventOut = node.child("EventOut")) {
            m_RegimeThresholds.addTriggerCode(currentRegimeID, srcPort + " > 0");
        }
        else {
             throw std::runtime_error("GeNN only supports neuron models which either emit events or perform state assignement on incomign events");
        }

        // If this event results in a regime change
        if(currentRegimeID != targetRegimeID) {
            throw std::runtime_error("GeNN does not support neuron models which change state based on incoming events");
        }
    }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void writeStateAssignement(const std::string &variableName, const std::string &multiplier, const std::string &srcPort)
    {
        m_CodeStream << variableName << " += " << multiplier << " * " << srcPort << ";" << std::endl;
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    SpineMLGenerator::CodeStream &m_CodeStream;
    RegimeThresholds &m_RegimeThresholds;
};
}   // Anonymous namespace

//----------------------------------------------------------------------------
// SpineMLGenerator::NeuronModel
//----------------------------------------------------------------------------
const char *SpineMLGenerator::NeuronModel::componentClassName = "neuron_body";
//----------------------------------------------------------------------------
SpineMLGenerator::NeuronModel::NeuronModel(const ModelParams::Neuron &params, const pugi::xml_node &componentClass)
{
    // Read aliases
    Aliases aliases(componentClass);

    // Loop through send ports
    LOGD_SPINEML << "\t\tSend ports:";
    std::unordered_set<std::string> sendPortAliases;
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();
        if(nodeType == "AnalogSendPort") {
            // If there is an alias matching this port name
            if(aliases.isAlias(portName)) {
                LOGD_SPINEML << "\t\t\tImplementing analogue send port '" << portName << "' as an alias";

                // Add it to the list of send port aliases
                sendPortAliases.insert(portName);
            }
            else {
                LOGD_SPINEML << "\t\t\tImplementing analogue send port '" << portName << "' using a GeNN model variable";
            }

            // Add send port to set
            m_SendPortVariables.insert(portName);
        }
        else if(nodeType == "EventSendPort") {
            if(m_SendPortSpike.empty()) {
                LOGD_SPINEML << "\t\t\tImplementing event send port '" << portName << "' as a GeNN spike";
                m_SendPortSpike = portName;
            }
            else {
                LOGD_SPINEML << "\t\t\tImplementing event send port '" << portName << "' as a GeNN spike-like-event";
                m_SendPortSpikeLikeEvent = portName;
                throw std::runtime_error("Spike-like event sending not currently implemented");
            }
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' send ports in neuron models");
        }
    }

    // Check that there are no unhandled receive ports
    LOGD_SPINEML << "\t\tReceive ports:";
    std::string trueSpikeReceivePort;
    std::vector<std::string> externalInputPorts;
    for(auto receivePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReceivePort").c_str())) {
        std::string nodeType = receivePort.node().name();
        const char *portName = receivePort.node().attribute("name").value();

        if(nodeType == "AnalogReceivePort") {
            if(params.isInputPortExternal(portName)) {
                LOGD_SPINEML << "\t\t\tImplementing analogue receive port '" << portName << "' as state variable to receive external input";
                externalInputPorts.emplace_back(portName);
            }
            else {
                LOGD_SPINEML << "\t\t\tImplementing analogue receive port '" << portName << "' as GeNN additional input variable";
                m_AdditionalInputVars.push_back({portName, "scalar", 0.0});
            }
        }
        else if(nodeType == "EventReceivePort") {
            LOGD_SPINEML << "\t\t\tImplementing event receive port '" << portName << "' as a GeNN additional input variable";
            m_AdditionalInputVars.push_back({portName, "scalar", 0.0});
            trueSpikeReceivePort = portName;
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' receive ports in neuron models");
        }
    }

    // Loop through reduce ports
    LOGD_SPINEML << "\t\tReduce ports:";
    for(auto reducePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReducePort").c_str())) {
        std::string nodeType = reducePort.node().name();
        const char *portName = reducePort.node().attribute("name").value();

        // **TODO** implement other reduce operations
        if(nodeType == "AnalogReducePort" && strcmp(reducePort.node().attribute("reduce_op").value(), "+") == 0) {
            if(params.isInputPortExternal(portName)) {
                LOGD_SPINEML << "\t\t\tImplementing analogue reduce port '" << portName << "' as state variable to receive external input";
                externalInputPorts.push_back(portName);
            }
            else {
                LOGD_SPINEML << "\t\t\tImplementing analogue reduce port '" << portName << "' as GeNN additional input variable";
                m_AdditionalInputVars.push_back({portName, "scalar", 0.0});
            }
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' reduce ports in neuron models");
        }
    }

    // Create a code stream for generating sim code
    CodeStream simCodeStream;

    // Create lambda function to end regime on all code streams when required
    auto regimeEndFunc =
        [&simCodeStream]
        (bool multipleRegimes, unsigned int currentRegimeID)
        {
            simCodeStream.onRegimeEnd(multipleRegimes, currentRegimeID);
        };

    // Generate model code using specified condition handler
    RegimeThresholds regimeThresholds;
    bool multipleRegimes;
    ObjectHandlerCondition objectHandlerCondition(simCodeStream,  m_SendPortSpike, regimeThresholds);
    ObjectHandlerEvent objectHandlerTrueSpike(simCodeStream, regimeThresholds);
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(simCodeStream);
    std::tie(multipleRegimes, m_InitialRegimeID) = generateModelCode(componentClass,
                                                                     {
                                                                         {trueSpikeReceivePort, &objectHandlerTrueSpike},
                                                                     },
                                                                     &objectHandlerCondition,
                                                                     {},
                                                                     &objectHandlerTimeDerivative,
                                                                     regimeEndFunc);

    auto variableParams = params.getVariableParams();

    // Loop through send ports which send an alias
    if(!sendPortAliases.empty()) {
        simCodeStream << "// Send port aliases" << std::endl;
    }
    for(const auto &s : sendPortAliases) {
        // Add simulation code to calculate send port value and store in state variable
        simCodeStream << s << " = " << aliases.getAliasCode(s) << ";" << std::endl;

        // Add this state variable to variable params set
        variableParams.insert(s);
    }

    // Add state variables for external input ports
    for(const auto &p : externalInputPorts) {
        variableParams.insert(p);
    }

    // Store generated code in class
    m_SimCode = simCodeStream.str();
    m_ThresholdConditionCode = regimeThresholds.getThresholdCode(multipleRegimes);

    // Generate aliases required for sim code and threshold
    std::stringstream aliasStream;
    aliases.genAliases(aliasStream, {m_SimCode, m_ThresholdConditionCode}, sendPortAliases);

    // Prepend simcode with aliases
    m_SimCode = aliasStream.str() + m_SimCode;

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, variableParams, multipleRegimes);

    // Add any derived parameters required for time-derivative
    objectHandlerTimeDerivative.addDerivedParams(m_ParamNames, m_DerivedParams);

    // Correctly wrap references to parameters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars, m_DerivedParams,
                             {&m_SimCode, &m_ThresholdConditionCode});
}
