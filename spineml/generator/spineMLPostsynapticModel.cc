#include "spineMLPostsynapticModel.h"

// Standard C++ includes
#include <iostream>
#include <regex>

// pugixml includes
#include "pugixml/pugixml.hpp"

// Spine ML generator includes
#include "modelParams.h"
#include "objectHandler.h"

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
SpineMLGenerator::SpineMLPostsynapticModel::SpineMLPostsynapticModel(const ModelParams::Postsynaptic &params)
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
    ObjectHandler::Error objectHandlerError;
    ObjectHandler::Condition objectHandlerCondition(decayCodeStream);
    ObjectHandlerImpulse objectHandlerImpulse;
    ObjectHandler::TimeDerivative objectHandlerTimeDerivative(decayCodeStream);
    const bool multipleRegimes = generateModelCode(componentClass, objectHandlerError,
                                                   objectHandlerCondition, objectHandlerImpulse,
                                                   objectHandlerTimeDerivative,
                                                   regimeEndFunc);

    // Store generated code in class
    m_DecayCode = decayCodeStream.str();

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, params.getVariableParams(), multipleRegimes);

    // Find names of analogue receive ports
    auto analogueReceivePortNames = findAnalogueReceivePortNames(componentClass);

    // Postsynaptic models use an analogue send port to transmit to the neuron
    auto analogueSendPorts = componentClass.children("AnalogSendPort");
    const size_t numAnalogueSendPorts = std::distance(analogueSendPorts.begin(), analogueSendPorts.end());
    if(numAnalogueSendPorts == 1) {
        const auto *analogueSendPortName = analogueSendPorts.begin()->attribute("name").value();
        std::cout << "\t\tAnalogue send port:" << analogueSendPortName << std::endl;

        // If this send port corresponds to a state variable
        auto correspondingVar = std::find_if(m_Vars.begin(), m_Vars.end(),
                                             [analogueSendPortName](const std::pair<std::string, std::string> &v)
                                             {
                                                 return (v.first == analogueSendPortName);
                                             });
        if(correspondingVar != m_Vars.end()) {
            // Set current converter code to simply return this variable
            m_CurrentConverterCode = correspondingVar->first;
        }
        // Otherwise
        else {
            // Search for an alias representing the analogue send port
            pugi::xpath_variable_set aliasSearchVars;
            aliasSearchVars.set("aliasName", analogueSendPortName);
            auto alias = componentClass.select_node("Dynamics/Alias[@name=$aliasName]", &aliasSearchVars);

            // If alias is found use it for current converter code
            if(alias) {
                m_CurrentConverterCode = alias.node().child_value("MathInline");
            }
            // Otherwise throw exception
            else {
                throw std::runtime_error("Cannot find alias:" + std::string(analogueSendPortName));
            }
        }
    }
    // Otherwise, throw an exception
    else if(numAnalogueSendPorts > 1) {
        throw std::runtime_error("GeNN postsynapses always output a single current");
    }

    // Check we only have one state variable
    // **NOTE** it could be possible to figure out what variable to replace with inSyn
    if(m_Vars.size() != 1) {
        throw std::runtime_error("GeNN currently only supports postsynaptic models with a single state variable");
    }
    else {
        // Substitute name of analogue send port for internal variable
        wrapAndReplaceVariableNames(m_DecayCode, m_Vars.front().first, "inSyn");
        wrapAndReplaceVariableNames(m_CurrentConverterCode, m_Vars.front().first, "inSyn");
        m_Vars.clear();
    }

    // Correctly wrap references to parameters, variables and analogue receive ports in code strings
    substituteModelVariables(m_ParamNames, m_Vars, analogueReceivePortNames,
                             {&m_DecayCode, &m_CurrentConverterCode});

    //std::cout << "DECAY CODE:" << std::endl << m_DecayCode << std::endl;
    //std::cout << "CURRENT CONVERTER CODE:" << std::endl << m_CurrentConverterCode << std::endl;
}