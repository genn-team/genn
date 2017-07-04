#include "weightUpdateModel.h"

// Standard C++ includes
#include <iostream>
#include <regex>

// pugixml includes
#include "pugixml/pugixml.hpp"

// SpineML common includes
#include "spineMLUtils.h"

// Spine ML generator includes
#include "modelParams.h"
#include "objectHandler.h"
#include "neuronModel.h"

using namespace SpineMLCommon;

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// ObjectHandlerEvent
//----------------------------------------------------------------------------
class ObjectHandlerEvent : public SpineMLGenerator::ObjectHandler::Base
{
public:
    ObjectHandlerEvent(SpineMLGenerator::CodeStream &codeStream) : m_CodeStream(codeStream){}

    //------------------------------------------------------------------------
    // ObjectHandler::Base virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID)
    {
        // If this event handler outputs an impulse
        auto outgoingImpulses = node.children("ImpulseOut");
        const size_t numOutgoingImpulses = std::distance(outgoingImpulses.begin(), outgoingImpulses.end());
        if(numOutgoingImpulses == 1) {
            m_CodeStream << "addtoinSyn = " << outgoingImpulses.begin()->attribute("port").value() << ";" << std::endl;
            m_CodeStream << "updatelinsyn;" << std::endl;
        }
        // Otherwise, throw an exception
        else if(numOutgoingImpulses > 1) {
            throw std::runtime_error("GeNN weigh updates always output a single impulse");
        }

        // Loop through state assignements
        for(auto stateAssign : node.children("StateAssignment")) {
            m_CodeStream << stateAssign.attribute("variable").value() << " = " << stateAssign.child_value("MathInline") << ";" << std::endl;
        }

        // If this condition results in a regime change
        if(currentRegimeID != targetRegimeID) {
            m_CodeStream << "_regimeID = " << targetRegimeID << ";" << std::endl;
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    SpineMLGenerator::CodeStream &m_CodeStream;
};
}


//----------------------------------------------------------------------------
// SpineMLGenerator::WeightUpdateModel
//----------------------------------------------------------------------------
SpineMLGenerator::WeightUpdateModel::WeightUpdateModel(const ModelParams::WeightUpdate &params,
                                                       const NeuronModel *srcNeuronModel,
                                                       const NeuronModel *trgNeuronModel)
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
    if(!componentClass || strcmp(componentClass.attribute("type").value(), "weight_update") != 0) {
        throw std::runtime_error("XML file:" + params.getURL() + " is not a SpineML 'weight_update' component - "
                                 "it's ComponentClass node is either missing or of the incorrect type");
    }

    // Loop through send ports
    std::cout << "\t\tSend ports:" << std::endl;
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();

        if(nodeType == "ImpulseSendPort") {
            if(m_SendPortSpikeImpulse.empty()) {
                std::cout << "\t\t\tImplementing impulse send port '" << portName << "' as a GeNN spike impulse" << std::endl;
                m_SendPortSpikeImpulse = portName;
            }
            else {
                throw std::runtime_error("GeNN weight update models only support a single spike impulse port");
            }
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' send ports in weight update models");
        }
    }

    // Loop through receive ports
    std::cout << "\t\tReceive ports:" << std::endl;
    std::string trueSpikeReceivePort;
    std::string spikeLikeEventReceivePort;
    std::map<std::string, std::string> receivePortVariableMap;
    for(auto receivePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReceivePort").c_str())) {
        std::string nodeType = receivePort.node().name();
        const char *portName = receivePort.node().attribute("name").value();
        const auto &portSrc = params.getPortSrc(portName);

        // If this port is an analogue receive port for some sort of postsynaptic neuron state variable
        if(nodeType == "EventReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON && srcNeuronModel->getSendPortSpike() == portSrc.second) {
            std::cout << "\t\t\tImplementing event receive port '" << portName << "' as GeNN true spike" << std::endl;
            trueSpikeReceivePort = portName;
        }
        // Otherwise if this port is an impulse receive port which receives spike impulses from weight update model
        else if(nodeType == "EventReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON && srcNeuronModel->getSendPortSpikeLikeEvent() == portSrc.second) {
            std::cout << "\t\t\tImplementing impulse receive port '" << portName << "' as GeNN spike-like event" << std::endl;
            spikeLikeEventReceivePort = portName;
        }
        // If this is an analog receive port from the presynaptic neuron, add send port variable to map with _pre suffix
        else if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON && srcNeuronModel->hasSendPortVariable(portSrc.second)) {
            std::cout << "\t\t\tImplementing analogue receive port '" << portName << "' using presynaptic neuron send port variable '" << portSrc.second << "'" << std::endl;
            receivePortVariableMap.insert(std::make_pair(portName, portSrc.second + "_pre"));
        }
        // If this is an analog receive port from the postsynaptic neuron, add send port variable to map with _post suffix
        else if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON && trgNeuronModel->hasSendPortVariable(portSrc.second)) {
            std::cout << "\t\t\tImplementing analogue receive port '" << portName << "' using postsynaptic neuron send port variable '" << portSrc.second << "'" << std::endl;
            receivePortVariableMap.insert(std::make_pair(portName, portSrc.second + "_post"));
        }
        else {
            throw std::runtime_error("GeNN does not currently support '" + nodeType + "' receive ports in weight update models");
        }
    }


    // Create code streams for generating sim and synapse dynamics code
    CodeStream simCodeStream;
    CodeStream synapseDynamicsStream;

    // Create lambda function to end regime on all code streams when required
    auto regimeEndFunc =
        [&simCodeStream, &synapseDynamicsStream](bool multipleRegimes, unsigned int currentRegimeID)
        {
            simCodeStream.onRegimeEnd(multipleRegimes, currentRegimeID);
            synapseDynamicsStream.onRegimeEnd(multipleRegimes, currentRegimeID);
        };

    // Generate model code using specified condition handler
    ObjectHandler::Condition objectHandlerCondition(synapseDynamicsStream);
    ObjectHandlerEvent objectHandlerTrueSpike(simCodeStream);
    ObjectHandlerEvent objectHandlerSpikeLikeEvent(simCodeStream);
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(synapseDynamicsStream);
    const bool multipleRegimes = generateModelCode(componentClass,
                                                   {
                                                       {trueSpikeReceivePort, &objectHandlerTrueSpike},
                                                       {spikeLikeEventReceivePort, &objectHandlerSpikeLikeEvent}
                                                   },
                                                   &objectHandlerCondition, {}, &objectHandlerTimeDerivative,
                                                   regimeEndFunc);

    // Store generated code in class
    m_SimCode = simCodeStream.str();
    m_SynapseDynamicsCode = synapseDynamicsStream.str();

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, params.getVariableParams(), multipleRegimes);

    // Wrap internal variables used in sim code
    wrapVariableNames(m_SimCode, "addtoinSyn");
    wrapVariableNames(m_SimCode, "updatelinsyn");

    // Correctly wrap and replace references to receive port variable in code string
    for(const auto &r : receivePortVariableMap) {
        wrapAndReplaceVariableNames(m_SimCode, r.first, r.second);
        wrapAndReplaceVariableNames(m_SynapseDynamicsCode, r.first, r.second);
    }

    // Correctly wrap references to paramters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars, {&m_SimCode, &m_SynapseDynamicsCode});

    //std::cout << "SIM CODE:" << std::endl << m_SimCode << std::endl;
    //std::cout << "SYNAPSE DYNAMICS CODE:" << std::endl << m_SynapseDynamicsCode << std::endl;
}