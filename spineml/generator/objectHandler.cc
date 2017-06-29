#include "objectHandler.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandler::Condition
//------------------------------------------------------------------------
void SpineMLGenerator::ObjectHandler::Condition::onObject(const pugi::xml_node &node,
                                                          unsigned int currentRegimeID, unsigned int targetRegimeID)
{
    // Get triggering code
    auto triggerCode = node.child("Trigger").child("MathInline");
    if(!triggerCode) {
        throw std::runtime_error("No trigger condition for transition between regimes");
    }

    // Write trigger condition
    m_CodeStream << "if(" << triggerCode.text().get() << ")" << CodeStream::OB(2);

    // Loop through state assignements
    for(auto stateAssign : node.children("StateAssignment")) {
        m_CodeStream << stateAssign.attribute("variable").value() << " = " << stateAssign.child_value("MathInline") << ";" << std::endl;
    }

    // If this condition results in a regime change
    if(currentRegimeID != targetRegimeID) {
        m_CodeStream << "_regimeID = " << targetRegimeID << ";" << std::endl;
    }

    // End of trigger condition
    m_CodeStream << CodeStream::CB(2);
}

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandler::TimeDerivative
//------------------------------------------------------------------------
void SpineMLGenerator::ObjectHandler::TimeDerivative::onObject(const pugi::xml_node &node,
                                                               unsigned int, unsigned int)
{
    // **TODO** identify cases where Euler is REALLY stupid
    m_CodeStream << node.attribute("variable").value() << " += DT * (" << node.child_value("MathInline") << ");" << std::endl;
}