#include "objectHandlerTimeDerivative.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandlerTimeDerivative
//------------------------------------------------------------------------
void SpineMLGenerator::ObjectHandlerTimeDerivative::onObject(const pugi::xml_node &node,
                                                             unsigned int, unsigned int)
{
    // **TODO** identify cases where Euler is REALLY stupid
    m_CodeStream << node.attribute("variable").value() << " += DT * (" << node.child_value("MathInline") << ");" << m_CodeStream.endl();
}