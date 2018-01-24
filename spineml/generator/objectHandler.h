#pragma once

// Standard C++ includes
#include <map>
#include <string>

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
    Condition(CodeStream &codeStream, const std::map<std::string, std::string> &aliases) : m_CodeStream(codeStream), m_Aliases(aliases){}

    //------------------------------------------------------------------------
    // ObjectHandler virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID) override;

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const std::map<std::string, std::string> &getAliases() const{ return m_Aliases; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    CodeStream &m_CodeStream;
    const std::map<std::string, std::string> &m_Aliases;
};

//------------------------------------------------------------------------
// SpineMLGenerator::ObjectHandler::TimeDerivative
//------------------------------------------------------------------------
class TimeDerivative : public Base
{
public:
    TimeDerivative(CodeStream &codeStream, const std::map<std::string, std::string> &aliases)
        : m_CodeStream(codeStream), m_Aliases(aliases){}

    //------------------------------------------------------------------------
    // ObjectHandler virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID) override;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void addDerivedParams(const NewModels::Base::StringVec &paramNames,
                          NewModels::Base::DerivedParamVec &derivedParams) const;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::string m_ClosedFormTauParamName;
    CodeStream &m_CodeStream;
    const std::map<std::string, std::string> &m_Aliases;
};
}   // namespace ObjectHandler
}   // namespace SpineMLGenerator