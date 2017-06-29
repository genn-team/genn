#include "neuronModel.h"

// Standard C++ includes
#include <iostream>
#include <sstream>

// Standard C includes
#include <cstring>

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML common includes
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
// ObjectHandlerNeuronCondition
//----------------------------------------------------------------------------
class ObjectHandlerNeuronCondition : public SpineMLGenerator::ObjectHandler::Condition
{
public:
    ObjectHandlerNeuronCondition(SpineMLGenerator::CodeStream &codeStream, const std::string &sendPortSpike)
        : SpineMLGenerator::ObjectHandler::Condition(codeStream), m_SendPortSpike(sendPortSpike){}

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
            // If there are existing threshold conditions, OR them with this one
            if(m_ThresholdCodeStream.tellp() > 0) {
                m_ThresholdCodeStream << " || ";
            }

            // Write trigger condition AND regime to threshold condition
            auto triggerCode = node.child("Trigger").child("MathInline");
            m_ThresholdCodeStream << "(_regimeID == " << currentRegimeID << " && (" << triggerCode.text().get() << "))";
        }
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::string getThresholdCode() const{ return m_ThresholdCodeStream.str(); }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ostringstream m_ThresholdCodeStream;
    std::string m_SendPortSpike;
};
}

//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLNeuronModel
//----------------------------------------------------------------------------
SpineMLGenerator::SpineMLNeuronModel::SpineMLNeuronModel(const ModelParams::Neuron &params)
{
    // Load XML document
    pugi::xml_document doc;
    auto result = doc.load_file(params.getURL().c_str());
    if(!result) {
        throw std::runtime_error("Could not open file:" + params.getURL() + ", error:" + result.description());
    }

    // Get SpineML root
    auto spineML = doc.child("SpineML");
    if(!spineML) {
        throw std::runtime_error("XML file:" + params.getURL() + " is not a SpineML component - it has no root SpineML node");
    }

    // Get component class
    auto componentClass = spineML.child("ComponentClass");
    if(!componentClass || strcmp(componentClass.attribute("type").value(), "neuron_body") != 0) {
        throw std::runtime_error("XML file:" + params.getURL() + " is not a SpineML 'neuron_body' component - "
                                 "it's ComponentClass node is either missing or of the incorrect type");
    }

    // Loop through send ports
    std::cout << "\t\tSend ports:" << std::endl;
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();
        if(nodeType == "AnalogSendPort") {
            std::cout << "\t\t\tImplementing analogue send port '" << portName << "' using a GeNN model variable" << std::endl;
            m_SendPortVariables.insert(portName);
        }
        else if(nodeType == "EventSendPort") {
            if(m_SendPortSpike.empty()) {
                std::cout << "\t\t\tImplementing event send port '" << portName << "' as a GeNN spike" << std::endl;
                m_SendPortSpike = portName;
            }
            else {
                std::cout << "\t\t\tImplementing event send port '" << portName << "' as a GeNN spike-like-event" << std::endl;
                m_SendPortSpikeLikeEvent = portName;
                throw std::runtime_error("Spike-like event sending not currently implemented");
            }
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' send ports in neuron models");
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
    ObjectHandlerNeuronCondition objectHandlerCondition(simCodeStream, m_SendPortSpike);
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(simCodeStream);
    const bool multipleRegimes = generateModelCode(componentClass, {}, &objectHandlerCondition,
                                                   {}, &objectHandlerTimeDerivative,
                                                   regimeEndFunc);

    // Store generated code in class
    m_SimCode = simCodeStream.str();
    m_ThresholdConditionCode = objectHandlerCondition.getThresholdCode();

    // Build the final vectors of parameter names and variables from model and
    // correctly wrap references to them in newly-generated code strings
    tie(m_ParamNames, m_Vars) = processModelVariables(componentClass, params.getVariableParams(),
                                                      multipleRegimes, {&m_SimCode, &m_ThresholdConditionCode});

    // If there is an analogue reduce port using the addition operator, it's probably a synaptic input current
    auto linearReducePorts = componentClass.select_nodes("AnalogReducePort[@reduce_op='+']");
    if(linearReducePorts.size() == 1) {
        const auto *linearReducePortName = linearReducePorts.first().node().attribute("name").value();
        wrapAndReplaceVariableNames(m_SimCode, linearReducePortName, "Isyn");
        wrapAndReplaceVariableNames(m_ThresholdConditionCode, linearReducePortName, "Isyn");
    }
    // Otherwise, throw an exception
    else if(linearReducePorts.size() > 1) {
        // **TODO** 'Alias' nodes in dynamics may be used to combine these together
        throw std::runtime_error("GeNN doesn't support multiple input currents going into neuron");
    }


    //std::cout << "SIM CODE:" << std::endl << m_SimCode << std::endl;
    //std::cout << "THRESHOLD CONDITION CODE:" << std::endl << m_ThresholdConditionCode << std::endl;
}