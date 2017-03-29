#pragma once

// Standard includes
#include <functional>
#include <map>
#include <string>
#include <vector>

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "CodeHelper.h"

// Forward declarations
namespace NewModels
{
    class Base;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::ParamValues
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
class ParamValues
{
public:
    ParamValues(const std::map<std::string, double> &values, const NewModels::Base &model)
        : m_Values(values), m_Model(model){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::vector<double> getValues() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const std::map<std::string, double> &m_Values;
    const NewModels::Base &m_Model;
};

//------------------------------------------------------------------------
// VarValues
//------------------------------------------------------------------------
class VarValues
{
public:
    VarValues(const std::map<std::string, double> &values, const NewModels::Base &model)
        : m_Values(values), m_Model(model){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::vector<double> getValues() const;

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const std::map<std::string, double> &m_Values;
    const NewModels::Base &m_Model;
};

//------------------------------------------------------------------------
// CodeStream
//------------------------------------------------------------------------
class CodeStream
{
public:
    CodeStream() : m_FirstNonEmptyRegime(true){}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void onRegimeEnd(bool multipleRegimes, unsigned int currentRegimeID);

    std::string str() const{ return m_CodeStream.str(); }

    std::string ob(unsigned int level)
    {
        return m_Helper.openBrace(level);
    }

    std::string cb(unsigned int level)
    {
        return m_Helper.closeBrace(level);
    }

    std::string endl() const
    {
        return m_Helper.endl();
    }

    //------------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------------
    //!< Operators to wrap underlying code stream curtesy of
    //!< http://stackoverflow.com/questions/25615253/correct-template-method-to-wrap-a-ostream
    template <typename T>
    CodeStream& operator << (T&& x) {
        m_CurrentRegimeCodeStream << std::forward<T>(x);
        return *this;
    }

    CodeStream& operator << (std::ostream& (*manip)(std::ostream&)) {
        m_CurrentRegimeCodeStream << manip;
        return *this;
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //!< GeNN code helper class to provide automatic bracketting
    //!< and intenting within this code stream
    CodeHelper m_Helper;

    //!< Reference to code stream that will be used to build
    //!< entire GeNN code string e.g. a block of sim code
    std::ostringstream m_CodeStream;

    //!< Flag used to determine whether this is the first regime written
    bool m_FirstNonEmptyRegime;

    //!< Internal codestream used to write regime code to
    std::ostringstream m_CurrentRegimeCodeStream;
};

//------------------------------------------------------------------------
// ObjectHandler
//------------------------------------------------------------------------
class ObjectHandler
{
public:
    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID) = 0;
};

//------------------------------------------------------------------------
// ObjectHandlerError
//------------------------------------------------------------------------
class ObjectHandlerError : public ObjectHandler
{
public:
    //------------------------------------------------------------------------
    // ObjectHandler virtuals
    //------------------------------------------------------------------------
    virtual void onObject(const pugi::xml_node &node, unsigned int currentRegimeID,
                          unsigned int targetRegimeID);
};

//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------
pugi::xml_node loadComponent(const std::string &url, const std::string &expectedType);

bool generateModelCode(const pugi::xml_node &componentClass, ObjectHandler &objectHandlerEvent,
                       ObjectHandler &objectHandlerCondition, ObjectHandler &objectHandlerImpulse,
                       ObjectHandler &objectHandlerTimeDerivative,
                       std::function<void(bool, unsigned int)> regimeEndFunc);
}   // namespace SpineMLGenerator