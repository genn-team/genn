#pragma once

// Standard C++ includes
#include <map>
#include <string>

// GeNN includes
#include "models.h"

// SpineML generator includes
#include "modelCommon.h"

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
    virtual ~Base(){}

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID) = 0;
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

private:
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

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addDerivedParams(const Models::Base::StringVec &paramNames,
                          Models::Base::DerivedParamVec &derivedParams) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_ClosedFormTauParamName;
    CodeStream &m_CodeStream;
};
}   // namespace ObjectHandler
}   // namespace SpineMLGenerator
