#include "spineMLWeightUpdateModel.h"

// Standard C++ includes
#include <iostream>
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
// ObjectHandlerEvent
//----------------------------------------------------------------------------
class ObjectHandlerEvent : public SpineMLGenerator::ObjectHandler
{
public:
    ObjectHandlerEvent(SpineMLGenerator::CodeStream &codeStream) : m_CodeStream(codeStream){}

    //------------------------------------------------------------------------
    // ObjectHandler virtuals
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
// SpineMLGenerator::SpineMLWeightUpdateModel
//----------------------------------------------------------------------------
SpineMLGenerator::SpineMLWeightUpdateModel::SpineMLWeightUpdateModel(const std::string &url,
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
    if(!componentClass || strcmp(componentClass.attribute("type").value(), "weight_update") != 0) {
        throw std::runtime_error("XML file:" + url + " is not a SpineML 'weight_update' component - "
                                 "it's ComponentClass node is either missing or of the incorrect type");
    }

    // Create code streams for generating sim and synapse dynamics code
    CodeStream simCodeStream;
    CodeStream synapseDynamicsStream;

    // Create lambda function to end regime on all code streams when required
    auto regimeEndFunc =
        [&simCodeStream, &synapseDynamicsStream]
        (bool multipleRegimes, unsigned int currentRegimeID)
        {
            simCodeStream.onRegimeEnd(multipleRegimes, currentRegimeID);
            synapseDynamicsStream.onRegimeEnd(multipleRegimes, currentRegimeID);
        };

    // Generate model code using specified condition handler
    ObjectHandlerError objectHandlerError;
    ObjectHandlerCondition objectHandlerCondition(synapseDynamicsStream);
    ObjectHandlerEvent objectHandlerEvent(simCodeStream);
    ObjectHandlerTimeDerivative objectHandlerTimeDerivative(synapseDynamicsStream);
    const bool multipleRegimes = generateModelCode(componentClass, objectHandlerEvent,
                                                   objectHandlerCondition, objectHandlerError,
                                                   objectHandlerTimeDerivative,
                                                   regimeEndFunc);

    // Store generated code in class
    m_SimCode = simCodeStream.str();
    m_SynapseDynamicsCode = synapseDynamicsStream.str();

    // Build the final vectors of parameter names and variables from model
    tie(m_ParamNames, m_Vars) = findModelVariables(componentClass, variableParams, multipleRegimes);

    // Wrap internal variables used in sim code
    wrapVariableNames(m_SimCode, "addtoinSyn");
    wrapVariableNames(m_SimCode, "updatelinsyn");

    // Correctly wrap references to paramters and variables in code strings
    substituteModelVariables(m_ParamNames, m_Vars, {&m_SimCode, &m_SynapseDynamicsCode});

    //std::cout << "SIM CODE:" << std::endl << m_SimCode << std::endl;
    //std::cout << "SYNAPSE DYNAMICS CODE:" << std::endl << m_SynapseDynamicsCode << std::endl;
}