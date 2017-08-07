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

        // Determine which variable impulse is being assigned to
        m_ImpulseAssignStateVar = stateAssigment.attribute("variable").value();

        // Match for A + B type expression with any amount of whitespace
        auto stateAssigmentCode = stateAssigment.child_value("MathInline");
        std::regex regex("\\s*([a-zA-Z_]+)\\s*\\+\\s*([a-zA-Z_]+)\\s*");
        std::cmatch match;
        if(std::regex_match(stateAssigmentCode, match, regex)) {
            // If match is successful check which of the variables being added 
            // match the impulse coming from the weight update 
            if(match[1].str() == m_SpikeImpulseReceivePort) {
                if(match[2].str() != m_ImpulseAssignStateVar) {
                    throw std::runtime_error("Mismatch between impulse handling expression and variable");
                }
                return;
            }
            else if(match[2].str() == m_SpikeImpulseReceivePort) {
                if(match[1].str() != m_ImpulseAssignStateVar) {
                    throw std::runtime_error("Mismatch between impulse handling expression and variable");
                }
                return;
            }
        }

        // If standard impulse code can't be identified, implement it as custom update lin syn code
        m_UpdateLinSynCode = stateAssigmentCode;
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    const std::string &getImpulseAssignStateVar() const{ return m_ImpulseAssignStateVar; }
    const std::string &getUpdateLinSynCode() const{ return m_UpdateLinSynCode; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::string m_SpikeImpulseReceivePort;
    std::string m_ImpulseAssignStateVar;
    std::string m_UpdateLinSynCode;
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
            const auto &neuronPortTrg = params.getOutputPortTrg(portName);
            if(neuronPortTrg.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON) {
                std::cout << "\t\t\tImplementing analogue send port '" << portName << "' using postsynaptic neuron additional input var '" << neuronPortTrg.second << "'" << std::endl;

                // Add mapping to vector
                sendPortVariables.push_back(std::make_pair(neuronPortTrg.second, portName));
            }
            else {
                throw std::runtime_error("GeNN does not support AnalogSendPorts which target anything other than postsynaptic neurons");
            }
        }
        else {
            throw std::runtime_error("GeNN does not support '" + nodeType + "' send ports in postsynaptic models");
        }
    }

    // Read aliases
    std::map<std::string, std::string> aliases;
    readAliases(componentClass, aliases);

    // Loop through receive ports
    std::cout << "\t\tReceive ports:" << std::endl;
    std::map<std::string, std::string> receivePortVariableMap;
    std::string spikeImpulseReceivePort;
    for(auto receivePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReceivePort").c_str())) {
        std::string nodeType = receivePort.node().name();
        const char *portName = receivePort.node().attribute("name").value();
        const auto &portSrc = params.getInputPortSrc(portName);

        // If this port is an analogue receive port for some sort of postsynaptic neuron state variable
        if(nodeType == "AnalogReceivePort" && portSrc.first == ModelParams::Base::PortSource::POSTSYNAPTIC_NEURON && trgNeuronModel->hasSendPortVariable(portSrc.second)) {
            std::cout << "\t\t\tImplementing analogue receive port '" << portName << "' using postsynaptic neuron send port variable '" << portSrc.second << "'" << std::endl;
            receivePortVariableMap.insert(std::make_pair(portName, portSrc.second));
        }
        // Otherwise if this port is an impulse receive port which receives spike impulses from weight update model
        else if(nodeType == "ImpulseReceivePort" && portSrc.first == ModelParams::Base::PortSource::WEIGHT_UPDATE && weightUpdateModel->getSendPortSpikeImpulse() == portSrc.second) {
            std::cout << "\t\t\tImplementing impulse receive port '" << portName << "' as GeNN weight update model input" << std::endl;
            spikeImpulseReceivePort = portName;
        }
        else {
            throw std::runtime_error("GeNN does not currently support '" + nodeType + "' receive ports in postsynaptic models");
        }
    }

    // Loop through reduce ports
    std::cout << "\t\tReduce ports:" << std::endl;
    std::string analogueReducePort;
    for(auto reducePort : componentClass.select_nodes(SpineMLUtils::xPathNodeHasSuffix("ReducePort").c_str())) {
        std::string nodeType = reducePort.node().name();
        const char *portName = reducePort.node().attribute("name").value();
        const auto &portSrc = params.getInputPortSrc(portName);

        // If this is an analogue reduce port which receives analogue input from weight update model
        if(nodeType == "AnalogReducePort" && portSrc.first == ModelParams::Base::PortSource::WEIGHT_UPDATE && weightUpdateModel->getSendPortAnalogue() == portSrc.second) {
            std::cout << "\t\t\tImplementing analogue reduce port '" << portName << "' as GeNN weight update model input" << std::endl;
            analogueReducePort = portName;
        }
        else {
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
    ObjectHandler::Condition objectHandlerCondition(decayCodeStream);
    ObjectHandlerImpulse objectHandlerImpulse(spikeImpulseReceivePort);
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(decayCodeStream);
    const bool multipleRegimes = generateModelCode(componentClass,
                                                   {}, &objectHandlerCondition,
                                                   {{spikeImpulseReceivePort, &objectHandlerImpulse}},
                                                   &objectHandlerTimeDerivative, regimeEndFunc);

    // Store generated code in class
    m_DecayCode = decayCodeStream.str();
    m_UpdateLinSynCode = objectHandlerImpulse.getUpdateLinSynCode();

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, params.getVariableParams(), multipleRegimes);

    // Determine what state variable inpulse is applied to
    const std::string &impulseAssignStateVar = objectHandlerImpulse.getImpulseAssignStateVar();

    // Loop through send port variables and build apply input code to update them
    for(const auto s : sendPortVariables) {
        // Resolve the alias used to get input value to sent
        std::string inputCode = getSendPortCode(aliases, m_Vars, s.second);

        // If incoming impulse is being assigned to a state variable, substitute it within the input code
        // **NOTE** we do this here to avoid ambiguity between port names in the postsynaptic and neuron models
        if(!impulseAssignStateVar.empty()) {
            wrapAndReplaceVariableNames(inputCode, impulseAssignStateVar, "inSyn");

            // Add code string to apply input
            m_ApplyInputCode += s.first + " += " + inputCode + ";\n";
        }
        // If the weight update is applying analogue input directly, substitute the name port this is coming in from with inSyn
        else if(!analogueReducePort.empty()) {
            wrapAndReplaceVariableNames(inputCode, analogueReducePort, "inSyn");

            // Add code string to apply input and zero it
            // **NOTE** analogue send ports don't have decay dynamics so this is required so only this timestep's inputs are applied
            m_ApplyInputCode += s.first + " += " + inputCode + "; $(inSyn) = 0;\n";

            if(!m_DecayCode.empty()) {
                throw std::runtime_error("Postsynaptic decay dynamics not supported when weight update provides continuous input");
            }
        }
    }

    // If incoming impulse is being assigned to a state variable
    if(!impulseAssignStateVar.empty()) {
        std::cout << "\t\tImpulse assign state variable:" << impulseAssignStateVar << std::endl;

        // Substitute name of analogue send port for internal variable
        wrapAndReplaceVariableNames(m_DecayCode, impulseAssignStateVar, "inSyn");

        // Substitute the name of the incoming impulse with $(addtoinSyn) and the state variable with $(inSyn)
        wrapAndReplaceVariableNames(m_UpdateLinSynCode, impulseAssignStateVar, "inSyn");
        wrapAndReplaceVariableNames(m_UpdateLinSynCode, spikeImpulseReceivePort, "addtoinSyn");

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
        wrapAndReplaceVariableNames(m_UpdateLinSynCode, r.first, r.second);
    }

    // Correctly wrap references to parameters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars,
                             {&m_DecayCode, &m_ApplyInputCode, &m_UpdateLinSynCode});
}
