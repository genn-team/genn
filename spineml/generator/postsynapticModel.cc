#include "postsynapticModel.h"

// Standard C++ includes
#include <algorithm>
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

    // Loop through send ports
    std::cout << "\t\tSend ports:" << std::endl;
    std::vector<std::pair<std::string, std::string>> sendPortVariables;
    for(auto sendPort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("SendPort").c_str())) {
        std::string nodeType = sendPort.node().name();
        const char *portName = sendPort.node().attribute("name").value();

        // If this is an analogue send port
        if(nodeType == "AnalogSendPort") {
            // Find the name of the port on the neuron which this send port targets
            std::string neuronPortTrg = params.getPortTrg(ModelParams::Base::PortSource::POSTSYNAPTIC_SYNAPSE, portName);
            std::cout << "\t\t\tImplementing analogue send port '" << portName << "' using postsynaptic neuron additional input var '" << neuronPortTrg << "'" << std::endl;

            // Add mapping to vector
            sendPortVariables.push_back(std::make_pair(neuronPortTrg, portName));
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

    // Loop through send port variables and build apply input code to update them
    for(const auto s : sendPortVariables) {
        m_ApplyInputCode += s.first + " += " + resolveAlias(componentClass, m_Vars, s.second) + ";\n";
    }

    // If incoming impulse is being assigned to a state variable
    const std::string &impulseAssignStateVar = objectHandlerImpulse.getImpulseAssignStateVar();
    if(!impulseAssignStateVar.empty()) {
        std::cout << "\t\tImpulse assign state variable:" << impulseAssignStateVar << std::endl;

        // Substitute name of analogue send port for internal variable
        wrapAndReplaceVariableNames(m_DecayCode, impulseAssignStateVar, "inSyn");
        wrapAndReplaceVariableNames(m_ApplyInputCode, impulseAssignStateVar, "inSyn");

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
        wrapAndReplaceVariableNames(m_ApplyInputCode, r.first, r.second);
    }

    // Correctly wrap references to parameters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars,
                             {&m_DecayCode, &m_ApplyInputCode});
}
