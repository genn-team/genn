#include "spineMLPostsynapticModel.h"

// Standard C++ includes
#include <regex>

// pugixml includes
#include "pugixml/pugixml.hpp"

// Spine ML generator includes
#include "objectHandlerCondition.h"
#include "objectHandlerTimeDerivative.h"


//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// ObjectHandlerImpulse
//----------------------------------------------------------------------------
class ObjectHandlerImpulse : public SpineMLGenerator::ObjectHandler
{
public:
    //----------------------------------------------------------------------------
    // SpineMLGenerator::ObjectHandlerCondition virtuals
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
SpineMLGenerator::SpineMLPostsynapticModel::SpineMLPostsynapticModel(const std::string &url,
                                                                     const std::set<std::string> &variableParams)
{
    // Load XML document
    pugi::xml_document doc;
    auto result = doc.load_file(url.c_str());
    if(!result) {
        throw std::runtime_error("Could not open file:" + url + ", error:" + result.description());
    }

    // Get SpineML root
    auto spineML = doc.child("SpineML");
    if(!spineML) {
        throw std::runtime_error("XML file:" + url + " is not a SpineML component - it has no root SpineML node");
    }

    // Get component class
    auto componentClass = spineML.child("ComponentClass");
    if(!componentClass || strcmp(componentClass.attribute("type").value(), "postsynapse") != 0) {
        throw std::runtime_error("XML file:" + url + " is not a SpineML 'postsynapse' component - "
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
    ObjectHandlerError objectHandlerError;
    ObjectHandlerCondition objectHandlerCondition(decayCodeStream);
    ObjectHandlerImpulse objectHandlerImpulse;
    ObjectHandlerTimeDerivative objectHandlerTimeDerivative(decayCodeStream);
    const bool multipleRegimes = generateModelCode(componentClass, objectHandlerError,
                                                   objectHandlerCondition, objectHandlerImpulse,
                                                   objectHandlerTimeDerivative,
                                                   regimeEndFunc);

    // Store generated code in class
    m_DecayCode = decayCodeStream.str();

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, variableParams, multipleRegimes);

    // Postsynaptic models use an analogue send port to transmit to the neuron
    auto analogueSendPorts = componentClass.children("AnalogSendPort");
    const size_t numAnalogueSendPorts = std::distance(analogueSendPorts.begin(), analogueSendPorts.end());
    if(numAnalogueSendPorts == 1) {
        const auto *analogueSendPortName = analogueSendPorts.begin()->attribute("name").value();
        std::cout << "Analogue send port:" << analogueSendPortName << std::endl;

        // If this send port corresponds to a state variable
        auto correspondingVar = std::find_if(m_Vars.begin(), m_Vars.end(),
                                             [analogueSendPortName](const std::pair<std::string, std::string> &v)
                                             {
                                                 return (v.first == analogueSendPortName);
                                             });
        if(correspondingVar != m_Vars.end()) {
            // Remove state variable as GeNN uses an internal variable for this
            m_Vars.erase(correspondingVar);

            // Set current converter code to simply return internal variable
            m_CurrentConverterCode = "$(inSyn)";

            // Substitute name of analogue send port for internal variable
            wrapAndReplaceVariableNames(m_DecayCode, analogueSendPortName, "inSyn");
        }
        // Otherwise
        else {
            // **TODO** follow back aliased variables
            throw std::runtime_error("GeNN doesn't currently support postsynaptic models with aliased output");
        }
    }
    // Otherwise, throw an exception
    else if(numAnalogueSendPorts > 1) {
        throw std::runtime_error("GeNN postsynapses always output a single current");
    }

    // Correctly wrap references to paramters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars, {&m_DecayCode, &m_CurrentConverterCode});

    std::cout << "DECAY CODE:" << std::endl << m_DecayCode << std::endl;
    std::cout << "CURRENT CONVERTER CODE:" << std::endl << m_CurrentConverterCode << std::endl;
}