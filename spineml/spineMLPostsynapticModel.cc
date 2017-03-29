#include "spineMLPostsynapticModel.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// Spine ML generator includes
#include "objectHandlerCondition.h"
#include "objectHandlerTimeDerivative.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::spineMLPostsynapticModel
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
    ObjectHandlerTimeDerivative objectHandlerTimeDerivative(decayCodeStream);
    const bool multipleRegimes = generateModelCode(componentClass, objectHandlerError,
                                                   objectHandlerCondition, objectHandlerError,
                                                   objectHandlerTimeDerivative,
                                                   regimeEndFunc);

    // Store generated code in class
    m_DecayCode = decayCodeStream.str();
    //m_CurrentConverterCode

    // Build the final vectors of parameter names and variables from model and
    // correctly wrap references to them in newly-generated code strings
    tie(m_ParamNames, m_Vars) = processModelVariables(componentClass, variableParams,
                                                      multipleRegimes, {&m_DecayCode, &m_CurrentConverterCode});


    std::cout << "DECAY CODE:" << std::endl << m_DecayCode << std::endl;
    std::cout << "CURRENT CONVERTER CODE:" << std::endl << m_CurrentConverterCode << std::endl;
}