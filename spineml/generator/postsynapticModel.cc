#include "postsynapticModel.h"

// Standard C++ includes
#include <algorithm>
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

        auto stateAssigment = node.child("StateAssignment");
        if(!stateAssigment) {
            throw std::runtime_error("GeNN only supports postsynaptic models where state assignment occurs");
        }

        // Match for A + B type expression with any amount of whitespace
        auto stateAssigmentCode = stateAssigment.child_value("MathInline");
        std::regex regex("\\s*([a-zA-Z_])+\\s*\\+\\s*([a-zA-Z_]+)\\s*");
        std::cmatch match;
        if(std::regex_match(stateAssigmentCode, match, regex)) {
            // If match is successful check which of the variables being added 
            // match the impulse coming from the weight update 
            if(match[1].str() == m_SpikeImpulseReceivePort) {
                m_ImpulseAssignStateVar = match[2].str();
                return;
            }
            else if(match[2].str() == m_SpikeImpulseReceivePort) {
                m_ImpulseAssignStateVar = match[1].str();
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
}


//----------------------------------------------------------------------------
// SpineMLGenerator::PostsynapticModel
//----------------------------------------------------------------------------
SpineMLGenerator::PostsynapticModel::PostsynapticModel(const ModelParams::Postsynaptic &params,
                                                       const NeuronModel *trgNeuronModel,
                                                       const WeightUpdateModel *weightUpdateModel)
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
    if(!componentClass || strcmp(componentClass.attribute("type").value(), "postsynapse") != 0) {
        throw std::runtime_error("XML file:" + params.getURL() + " is not a SpineML 'postsynapse' component - "
                                 "it's ComponentClass node is either missing or of the incorrect type");
    }

    // Search for source port in post synaptic update which targets the
    // analogue reduce port used for Isyn in the target neuron population
    const auto &trgNeuronReducePortSrc = params.getPortSrc(trgNeuronModel->getReducePortIsyn());

    // Loop through send ports
    std::cout << "\t\tSend ports:" << std::endl;
    std::string neuronInputSendPort;
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();

        if(nodeType == "AnalogSendPort") {
            if(neuronInputSendPort.empty() && trgNeuronReducePortSrc.first == ModelParams::Base::PortSource::POSTSYNAPTIC_SYNAPSE && trgNeuronReducePortSrc.second == portName) {
                std::cout << "\t\t\tImplementing analogue send port '" << portName << "' as a GeNN neuron input current" << std::endl;
                neuronInputSendPort = portName;
            }
            else {
                throw std::runtime_error("GeNN postsynaptic models only support a single analogue send port");
            }
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' send ports in postsynaptic models");
        }
    }

    // Loop through receive ports
    std::cout << "\t\tReceive ports:" << std::endl;
    std::map<std::string, std::string> receivePortVariableMap;
    std::string spikeImpulseReceivePort;
    for(auto receivePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReceivePort").c_str())) {
        std::string nodeType = receivePort.node().name();
        const char *portName = receivePort.node().attribute("name").value();
        const auto &portSrc = params.getPortSrc(portName);

        // If this port is an analogue receive port for some sort of postsynaptic neuron state variable
        if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON && trgNeuronModel->hasSendPortVariable(portSrc.second)) {
            std::cout << "\t\t\tImplementing analogue receive port '" << portName << "' using postsynaptic neuron send port variable '" << portSrc.second << "'" << std::endl;
            receivePortVariableMap.insert(std::make_pair(portName, portSrc.second));
        }
        // Otherwise if this port is an impulse receive port which receives spike impulses from weight update model
        else if(nodeType == "ImpulseReceivePort" && portSrc.first == ModelParams::Base::PortSource::WEIGHT_UPDATE && weightUpdateModel->getSendPortSpikeImpulse() == portSrc.second) {
            std::cout << "\t\t\tImplementing impulse receive port '" << portName << "' as GeNN spike impulse" << std::endl;
            spikeImpulseReceivePort = portName;
        }
        else {
            throw std::runtime_error("GeNN does not currently support '" + nodeType + "' receive ports in postsynaptic models");
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
    ObjectHandler::Condition objectHandlerCondition(decayCodeStream);
    ObjectHandlerImpulse objectHandlerImpulse(spikeImpulseReceivePort);
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(decayCodeStream);
    const bool multipleRegimes = generateModelCode(componentClass,
                                                   {}, &objectHandlerCondition,
                                                   {{spikeImpulseReceivePort, &objectHandlerImpulse}},
                                                   &objectHandlerTimeDerivative, regimeEndFunc);

    // Store generated code in class
    m_DecayCode = decayCodeStream.str();

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, params.getVariableParams(), multipleRegimes);

    // If an analogue send port has been assigned to provide neuron input
    if(!neuronInputSendPort.empty()) {
        std::cout << "\t\tAnalogue send port:" << neuronInputSendPort << std::endl;

        // If this send port corresponds to a state variable
        auto correspondingVar = std::find_if(m_Vars.begin(), m_Vars.end(),
                                             [neuronInputSendPort](const std::pair<std::string, std::string> &v)
                                             {
                                                 return (v.first == neuronInputSendPort);
                                             });
        if(correspondingVar != m_Vars.end()) {
            // Set current converter code to simply return this variable
            m_CurrentConverterCode = correspondingVar->first;
        }
        // Otherwise
        else {
            // Search for an alias representing the analogue send port
            pugi::xpath_variable_set aliasSearchVars;
            aliasSearchVars.set("aliasName", neuronInputSendPort.c_str());
            auto alias = componentClass.select_node("Dynamics/Alias[@name=$aliasName]", &aliasSearchVars);

            // If alias is found use it for current converter code
            if(alias) {
                m_CurrentConverterCode = alias.node().child_value("MathInline");
            }
            // Otherwise throw exception
            else {
                throw std::runtime_error("Cannot find alias:" + neuronInputSendPort);
            }
        }
    }

    // If incoming impulse is being assigned to a state variable
    const std::string &impulseAssignStateVar = objectHandlerImpulse.getImpulseAssignStateVar();
    if(!impulseAssignStateVar.empty()) {
        std::cout << "\t\tImpulse assign state variable:" << impulseAssignStateVar << std::endl;

        // Substitute name of analogue send port for internal variable
        wrapAndReplaceVariableNames(m_DecayCode, impulseAssignStateVar, "inSyn");
        wrapAndReplaceVariableNames(m_CurrentConverterCode, impulseAssignStateVar, "inSyn");

        // As this variable is being implemented using a built in GeNN state variable, remove it from variables
        auto stateVar = std::find_if(m_Vars.begin(), m_Vars.end(),
                                     [impulseAssignStateVar](const std::pair<std::string, std::string> &var)
                                     {
                                         return (var.first == impulseAssignStateVar);
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
        wrapAndReplaceVariableNames(m_CurrentConverterCode, r.first, r.second);
    }

    // Correctly wrap references to parameters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars,
                             {&m_DecayCode, &m_CurrentConverterCode});
}
