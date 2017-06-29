#include "spineMLPostsynapticModel.h"

// Standard C++ includes
#include <iostream>
#include <regex>

// pugixml includes
#include "pugixml/pugixml.hpp"

// Spine ML generator includes
#include "modelParams.h"
#include "objectHandler.h"
#include "spineMLNeuronModel.h"
#include "spineMLWeightUpdateModel.h"

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
            throw std::runtime_error("GeNN only supports postsynaptic models where something other than state assignment occurs");
        }

        auto stateAssigmentCode = stateAssigment.child_value("MathInline");
        std::regex regex("\\s*([a-zA-Z_])+\\s*\\+\\s*([a-zA-Z_]+)\\s*");
        if(!std::regex_match(stateAssigmentCode, regex)) {
            throw std::runtime_error("GeNN only supports postsynaptic models which add incoming weight to state variable");
        }

        // **TODO** check that variable is analogue send port
        // **TODO** check that one of matched variables is analogue send port and other is impulse receive port
    }
};
}


//----------------------------------------------------------------------------
// SpineMLGenerator::SpineMLPostsynapticModel
//----------------------------------------------------------------------------
SpineMLGenerator::SpineMLPostsynapticModel::SpineMLPostsynapticModel(const ModelParams::Postsynaptic &params,
                                                                     const SpineMLNeuronModel *trgNeuronModel,
                                                                     const SpineMLWeightUpdateModel *weightUpdateModel)
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
    // **YUCK** this is a gross way of testing name
    std::cout << "\t\tSend ports:" << std::endl;
    std::string neuronInputSendPort;
    for(auto node : componentClass.children()) {
        std::string nodeType = node.name();
        if(nodeType.size() > 8 && nodeType.substr(nodeType.size() - 8) == "SendPort") {
            const char *portName = node.attribute("name").value();
            if(nodeType == "AnalogSendPort") {
                if(neuronInputSendPort.empty()) {
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
    }

    // Loop through receive ports
    // **YUCK** this is a gross way of testing name
    std::cout << "\t\tReceive ports:" << std::endl;
    std::map<std::string, std::string> receivePortVariableMap;
    std::string spikeImpulseReceivePort;
    for(auto node : componentClass.children()) {
        std::string nodeType = node.name();
        if(nodeType.size() > 11 && nodeType.substr(nodeType.size() - 11) == "ReceivePort") {
            const char *portName = node.attribute("name").value();
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
    ObjectHandlerImpulse objectHandlerImpulse;
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

    // Check we only have one state variable
    // **NOTE** it could be possible to figure out what variable to replace with inSyn
    if(m_Vars.size() != 1) {
        throw std::runtime_error("GeNN currently only supports postsynaptic models with a single state variable");
    }
    else {
        // Subool hasSendPortVariable() const;bstitute name of analogue send port for internal variable
        wrapAndReplaceVariableNames(m_DecayCode, m_Vars.front().first, "inSyn");
        wrapAndReplaceVariableNames(m_CurrentConverterCode, m_Vars.front().first, "inSyn");
        m_Vars.clear();
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
