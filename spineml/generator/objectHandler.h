#pragma once

// SpineML generator includes
#include "spineMLModelCommon.h"

// Forward declarations
namespace pugi
{
    class xml_node;
}

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandler::Base
//------------------------------------------------------------------------
namespace SpineMLGenerator
{
namespace ObjectHandler
{
class Base
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID) = 0;
};

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandler::Error
//------------------------------------------------------------------------
class Error : public Base
{
public:
    //------------------------------------------------------------------------
    // ObjectHandler virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID);
};

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandler::Condition
//------------------------------------------------------------------------
class Condition : public Base
{
public:
    Condition(CodeStream &codeStream) : m_CodeStream(codeStream){}

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

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandler::TimeDerivative
//------------------------------------------------------------------------
class TimeDerivative : public Base
{
public:
    TimeDerivative(CodeStream &codeStream) : m_CodeStream(codeStream){}

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
}   // namespace ObjectHandler
}   // namespace SpineMLGenerator