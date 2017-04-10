#pragma once

// Standard C++ includes
#include <sstream>

// Spine ML generator includes
#include "spineMLModelCommon.h"

// Forward declarations
namespace pugi
{
    class xml_node;
}

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandlerTimeDerivative
//------------------------------------------------------------------------
namespace SpineMLGenerator
{
class ObjectHandlerTimeDerivative : public ObjectHandler
{
public:
    ObjectHandlerTimeDerivative(CodeStream &codeStream) : m_CodeStream(codeStream){}

    //------------------------------------------------------------------------
    // ObjectHandler virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID) override;

protected:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    CodeStream &m_CodeStream;
};
}   // namespace SpineMLGenerator