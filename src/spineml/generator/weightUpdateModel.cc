#include "weightUpdateModel.h"

// Standard C++ includes
#include <iostream>
#include <regex>

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
    ObjectHandlerEvent(SpineMLGenerator::CodeStream &codeStream, bool heterogeneousDelay) 
    :   m_CodeStream(codeStream), m_HeterogeneousDelay(heterogeneousDelay) 
    {}

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
            const std::string weightPort = outgoingImpulses.begin()->attribute("port").value();
            if(m_HeterogeneousDelay) {
                m_CodeStream << "$(addToInSynDelay, " << weightPort << ",  _delay);" << std::endl;
            }
            else {
                m_CodeStream << "$(addToInSyn, " << weightPort << ");" << std::endl;
            }
        }
        // Otherwise, throw an exception
        else if(numOutgoingImpulses > 1) {
            throw std::runtime_error("GeNN weigh updates always output a single impulse");
        }

        // If this event handler outputs an event
        auto outgoingEvents = node.children("EventOut");
        const size_t numOutgoingEvents = std::distance(outgoingEvents.begin(), outgoingEvents.end());
        if(numOutgoingEvents == 1) {
            if(m_HeterogeneousDelay) {
                m_CodeStream << "$(addToInSynDelay, 1.0,  _delay);" << std::endl;
            }
            else {
                m_CodeStream << "$(addToInSyn, 1.0);" << std::endl;
            }
        }
        // Otherwise, throw an exception
        else if(numOutgoingEvents > 1) {
            throw std::runtime_error("GeNN weigh updates always output a single event");
        }

        // Loop through state assignements
        for(auto stateAssign : node.children("StateAssignment")) {
            m_CodeStream << stateAssign.attribute("variable").value() << " = " << stateAssign.child_value("MathInline") << ";" << std::endl;
        }

        // If this event results in a regime change
        if(currentRegimeID != targetRegimeID) {
            m_CodeStream << "_regimeID = " << targetRegimeID << ";" << std::endl;
        }
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    SpineMLGenerator::CodeStream &m_CodeStream;
    const bool m_HeterogeneousDelay;
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
    // Are heterogeneous delays required?
    const bool heterogeneousDelay = (params.getMaxDendriticDelay() > 1);

    // Read aliases
    Aliases aliases(componentClass);

    // Loop through send ports
    LOGD_SPINEML << "\t\tSend ports:";
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();

        if(nodeType == "ImpulseSendPort" && m_SendPortSpikeImpulse.empty() && m_SendPortAnalogue.empty()) {
            LOGD_SPINEML << "\t\t\tImplementing impulse send port '" << portName << "' as a GeNN linear synapse";
            m_SendPortSpikeImpulse = portName;
        }
        else if(nodeType == "AnalogSendPort" && m_SendPortSpikeImpulse.empty() && m_SendPortAnalogue.empty()) {
            LOGD_SPINEML << "\t\t\tImplementing analogue send port '" << portName << "' as a GeNN linear synapse";

            // Mark this as the analogue send port
            m_SendPortAnalogue = portName;
        }
        else if(nodeType == "EventSendPort" && m_SendPortSpikeImpulse.empty() && m_SendPortAnalogue.empty()) {
            LOGD_SPINEML << "\t\t\tImplementing event send port '" << portName << "' as a GeNN linear synapse";
            m_SendPortSpikeImpulse = portName;
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' send ports in weight update models");
        }
    }

    // Loop through receive ports
    LOGD_SPINEML << "\t\tReceive ports:";
    std::string trueSpikeReceivePort;
    std::string spikeLikeEventReceivePort;
    std::map<std::string, std::string> receivePortVariableMap;
    for(auto receivePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReceivePort").c_str())) {
        std::string nodeType = receivePort.node().name();
        const char *portName = receivePort.node().attribute("name").value();
        const auto &portSrc = params.getInputPortSrc(portName);

        // If this port is an event receive port which receives events from presynaptic neuron
        // **NOTE** hard-coded spike sources are indicated by nullptr and emit spikes from hardcoded port "spike"
        if(nodeType == "EventReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON
            && ((srcNeuronModel == nullptr && portSrc.second == "spike") || srcNeuronModel->getSendPortSpike() == portSrc.second))
        {
            LOGD_SPINEML << "\t\t\tImplementing event receive port '" << portName << "' as GeNN true spike";
            trueSpikeReceivePort = portName;
        }
        // Otherwise if this port is an impulse receive port which receives spike impulses from weight update model
        else if(nodeType == "EventReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON
            && srcNeuronModel->getSendPortSpikeLikeEvent() == portSrc.second)
        {
            LOGD_SPINEML << "\t\t\tImplementing impulse receive port '" << portName << "' as GeNN spike-like event";
            spikeLikeEventReceivePort = portName;
        }
        // If this is an analog receive port from the presynaptic neuron, add send port variable to map with _pre suffix
        else if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::PRESYNAPTIC_NEURON
            && srcNeuronModel->hasSendPortVariable(portSrc.second))
        {
            LOGD_SPINEML << "\t\t\tImplementing analogue receive port '" << portName << "' using presynaptic neuron send port variable '" << portSrc.second << "'";
            receivePortVariableMap.emplace(portName, portSrc.second + "_pre");
        }
        // If this is an analog receive port from the postsynaptic neuron, add send port variable to map with _post suffix
        else if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON
            && trgNeuronModel->hasSendPortVariable(portSrc.second))
        {
            LOGD_SPINEML << "\t\t\tImplementing analogue receive port '" << portName << "' using postsynaptic neuron send port variable '" << portSrc.second << "'";
            receivePortVariableMap.emplace(portName, portSrc.second + "_post");
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
            LOGD_SPINEML << "\t\t\tImplementing analogue reduce port '" << portName << "' using presynaptic neuron send port variable '" << portSrc.second << "'";
            receivePortVariableMap.emplace(portName, portSrc.second + "_pre");
        }
        else if(nodeType == "AnalogReducePort" && portSrc.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON
            && trgNeuronModel->hasSendPortVariable(portSrc.second) && strcmp(reducePort.node().attribute("reduce_op").value(), "+") == 0)
        {
            LOGD_SPINEML << "\t\t\tImplementing analogue reduce port '" << portName << "' using postsynaptic neuron send port variable '" << portSrc.second << "'";
            receivePortVariableMap.emplace(portName, portSrc.second + "_post");
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
    bool multipleRegimes;
    ObjectHandler::Condition objectHandlerCondition(synapseDynamicsStream);
    ObjectHandlerEvent objectHandlerTrueSpike(simCodeStream, heterogeneousDelay);
    ObjectHandlerEvent objectHandlerSpikeLikeEvent(simCodeStream, heterogeneousDelay);
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(synapseDynamicsStream);
    std::tie(multipleRegimes, m_InitialRegimeID) = generateModelCode(componentClass,
                                                                     {
                                                                         {trueSpikeReceivePort, &objectHandlerTrueSpike},
                                                                         {spikeLikeEventReceivePort, &objectHandlerSpikeLikeEvent}
                                                                     },
                                                                     &objectHandlerCondition,
                                                                     {}, &objectHandlerTimeDerivative,
                                                                     regimeEndFunc);

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, params.getVariableParams(), multipleRegimes);

    // If model has heterogeneos delays, add 8-bit unsigned delay to vars
    if(heterogeneousDelay) {
        assert(params.getMaxDendriticDelay() < 0xFF);
        
        LOGD_SPINEML << "\t\tUsing uint8_t for dendritic delay";
        m_Vars.push_back({"_delay", "uint8_t"});
    }

    // Add any derived parameters required for time-derivative
    objectHandlerTimeDerivative.addDerivedParams(m_ParamNames, m_DerivedParams);

    // If we have an analogue send port
    std::unordered_set<std::string> excludeSynapseDynamicsAliases;
    if(!m_SendPortAnalogue.empty()) {
        // If it's an alias, use alias value as output
        // **NOTE** this will cause any dependencies of the aliases to be included
        if(aliases.isAlias(m_SendPortAnalogue)) {
            excludeSynapseDynamicsAliases.insert(m_SendPortAnalogue);
            if(heterogeneousDelay) {
                synapseDynamicsStream << "$(addToInSynDelay, "<< aliases.getAliasCode(m_SendPortAnalogue) << ", _delay);" << std::endl;
            }
            else {
                synapseDynamicsStream << "$(addToInSyn, " << aliases.getAliasCode(m_SendPortAnalogue) << ");" << std::endl;
            }
        }
        // Otherwise, just insert variable name, it'll be wrapped later
        else {
            if(heterogeneousDelay) {
                synapseDynamicsStream << "$(addToInSynDelay, "<< m_SendPortAnalogue << ", _delay);" << std::endl;
            }
            else {
                synapseDynamicsStream << "$(addToInSyn, " << m_SendPortAnalogue << ");" << std::endl;
            }
        }
    }

    // Store generated code in class
    m_SimCode = simCodeStream.str();
    m_SynapseDynamicsCode = synapseDynamicsStream.str();

    // Generate aliases required for sim code
    std::stringstream simCodeAliasStream;
    aliases.genAliases(simCodeAliasStream, {m_SimCode});
    m_SimCode = simCodeAliasStream.str() + m_SimCode;

    // Generate aliases required for sim code and threshold
    std::stringstream synapseDynamicsAliasStream;
    aliases.genAliases(synapseDynamicsAliasStream, {m_SynapseDynamicsCode}, excludeSynapseDynamicsAliases);
    m_SynapseDynamicsCode = synapseDynamicsAliasStream.str() + m_SynapseDynamicsCode;

    // Correctly wrap and replace references to receive port variable in code string
    for(const auto &r : receivePortVariableMap) {
        wrapAndReplaceVariableNames(m_SimCode, r.first, r.second);
        wrapAndReplaceVariableNames(m_SynapseDynamicsCode, r.first, r.second);
    }

    // Correctly wrap references to paramters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars, m_DerivedParams,
                             {&m_SimCode, &m_SynapseDynamicsCode});
}
