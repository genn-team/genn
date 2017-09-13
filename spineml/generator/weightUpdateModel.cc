#include "weightUpdateModel.h"

// Standard C++ includes
#include <iostream>
#include <regex>

// Standard C includes
#include <cstring>

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
const char *SpineMLGenerator::WeightUpdateModel::componentClassName = "weight_update";
//----------------------------------------------------------------------------
SpineMLGenerator::WeightUpdateModel::WeightUpdateModel(const ModelParams::WeightUpdate &params,
                                                       const pugi::xml_node &componentClass,
                                                       const NeuronModel *srcNeuronModel,
                                                       const NeuronModel *trgNeuronModel)
{
    // Read aliases
    std::map<std::string, std::string> aliases;
    readAliases(componentClass, aliases);

    // Loop through send ports
    std::cout << "\t\tSend ports:" << std::endl;
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();

        if(nodeType == "ImpulseSendPort" && m_SendPortSpikeImpulse.empty() && m_SendPortAnalogue.empty()) {
            std::cout << "\t\t\tImplementing impulse send port '" << portName << "' as a GeNN linear synapse" << std::endl;
            m_SendPortSpikeImpulse = portName;
        }
        else if(nodeType == "AnalogSendPort" && m_SendPortSpikeImpulse.empty() && m_SendPortAnalogue.empty()) {
            std::cout << "\t\t\tImplementing analogue send port '" << portName << "' as a GeNN linear synapse" << std::endl;
            m_SendPortAnalogue = portName;
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
        const auto &portSrc = params.getInputPortSrc(portName);

        // If this port is an analogue receive port for some sort of postsynaptic neuron state variable
        if(nodeType == "EventReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON
            && srcNeuronModel->getSendPortSpike() == portSrc.second)
        {
            std::cout << "\t\t\tImplementing event receive port '" << portName << "' as GeNN true spike" << std::endl;
            trueSpikeReceivePort = portName;
        }
        // Otherwise if this port is an impulse receive port which receives spike impulses from weight update model
        else if(nodeType == "EventReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON
            && srcNeuronModel->getSendPortSpikeLikeEvent() == portSrc.second)
        {
            std::cout << "\t\t\tImplementing impulse receive port '" << portName << "' as GeNN spike-like event" << std::endl;
            spikeLikeEventReceivePort = portName;
        }
        // If this is an analog receive port from the presynaptic neuron, add send port variable to map with _pre suffix
        else if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON
            && srcNeuronModel->hasSendPortVariable(portSrc.second))
        {
            std::cout << "\t\t\tImplementing analogue receive port '" << portName << "' using presynaptic neuron send port variable '" << portSrc.second << "'" << std::endl;
            receivePortVariableMap.insert(std::make_pair(portName, portSrc.second + "_pre"));
        }
        // If this is an analog receive port from the postsynaptic neuron, add send port variable to map with _post suffix
        else if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON
            && trgNeuronModel->hasSendPortVariable(portSrc.second))
        {
            std::cout << "\t\t\tImplementing analogue receive port '" << portName << "' using postsynaptic neuron send port variable '" << portSrc.second << "'" << std::endl;
            receivePortVariableMap.insert(std::make_pair(portName, portSrc.second + "_post"));
        }
        else {
            throw std::runtime_error("GeNN does not currently support '" + nodeType + "' receive ports in weight update models");
        }
    }

    // Check that there are no unhandled reduce ports
    for(auto reducePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReducePort").c_str())) {
        std::string nodeType = reducePort.node().name();
        const char *portName = reducePort.node().attribute("name").value();
        const auto &portSrc = params.getInputPortSrc(portName);

        if(nodeType == "AnalogReducePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON
            && srcNeuronModel->hasSendPortVariable(portSrc.second) && strcmp(reducePort.node().attribute("reduce_op").value(), "+") == 0)
        {
            std::cout << "\t\t\tImplementing analogue reduce port '" << portName << "' using presynaptic neuron send port variable '" << portSrc.second << "'" << std::endl;
            receivePortVariableMap.insert(std::make_pair(portName, portSrc.second + "_pre"));
        }
        else {
            throw std::runtime_error("GeNN does not currently support '" + std::string(reducePort.node().name()) + "' reduce ports in weight update models");
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
    ObjectHandler::Condition objectHandlerCondition(synapseDynamicsStream, aliases);
    ObjectHandlerEvent objectHandlerTrueSpike(simCodeStream);
    ObjectHandlerEvent objectHandlerSpikeLikeEvent(simCodeStream);
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(synapseDynamicsStream, aliases);
    const bool multipleRegimes = generateModelCode(componentClass,
                                                   {
                                                       {trueSpikeReceivePort, &objectHandlerTrueSpike},
                                                       {spikeLikeEventReceivePort, &objectHandlerSpikeLikeEvent}
                                                   },
                                                   &objectHandlerCondition, {}, &objectHandlerTimeDerivative,
                                                   regimeEndFunc);

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, params.getVariableParams(), multipleRegimes);

    // Add any derived parameters required for time-derivative
    objectHandlerTimeDerivative.addDerivedParams(m_ParamNames, m_DerivedParams);

    // If we have an analogue send port, add code to apply it to synapse dynamics
    if(!m_SendPortAnalogue.empty()) {
        synapseDynamicsStream << "addtoinSyn = " << getSendPortCode(aliases, m_Vars, m_SendPortAnalogue) << ";" << std::endl;
        synapseDynamicsStream << "updatelinsyn;" << std::endl;
    }

    // Store generated code in class
    m_SimCode = simCodeStream.str();
    m_SynapseDynamicsCode = synapseDynamicsStream.str();

    // Wrap internal variables used in sim code
    wrapVariableNames(m_SimCode, "addtoinSyn");
    wrapVariableNames(m_SimCode, "updatelinsyn");
    wrapVariableNames(m_SynapseDynamicsCode, "addtoinSyn");
    wrapVariableNames(m_SynapseDynamicsCode, "updatelinsyn");

    // Correctly wrap and replace references to receive port variable in code string
    for(const auto &r : receivePortVariableMap) {
        wrapAndReplaceVariableNames(m_SimCode, r.first, r.second);
        wrapAndReplaceVariableNames(m_SynapseDynamicsCode, r.first, r.second);
    }

    // Correctly wrap references to paramters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars, m_DerivedParams,
                             {&m_SimCode, &m_SynapseDynamicsCode});
}