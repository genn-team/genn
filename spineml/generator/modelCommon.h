#pragma once

// Standard includes
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "codeStream.h"
#include "newModels.h"

// Forward declarations
namespace SpineMLGenerator
{
    namespace ObjectHandler
    {
        class Base;
    }
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
template<typename M>
class VarValues
{
public:
    VarValues(const std::map<std::string, double> &values, const M &model)
        : m_Values(values), m_Model(model){}

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    std::vector<double> getValues() const
    {
        // Get variables from model
        auto modelVars = m_Model.getVars();

        // Reserve vector of values to match it
        std::vector<double> varValues;
        varValues.reserve(modelVars.size());

        // Populate this vector with either values from map or 0s
        std::transform(modelVars.begin(), modelVars.end(),
                       std::back_inserter(varValues),
                       [this](const std::pair<std::string, std::string> &n)
                       {
                           if(n.first == "_regimeID") {
                               return (double)m_Model.getInitialRegimeID();
                           }
                           else {
                               auto value = m_Values.find(n.first);
                               if(value == m_Values.end()) {
                                   return 0.0;
                               }
                               else {
                                   return value->second;
                               }
                           }
                        });
        return varValues;
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const std::map<std::string, double> &m_Values;
    const M &m_Model;
};

//------------------------------------------------------------------------
// CodeStream
//------------------------------------------------------------------------
class CodeStream : public ::CodeStream
{
public:
    CodeStream() : m_FirstNonEmptyRegime(true), m_CodeStream(m_Stream)
    {
        setSink(m_CurrentRegimeStream);
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void onRegimeEnd(bool multipleRegimes, unsigned int currentRegimeID);

    void flush();

    std::string str(){ flush(); return m_Stream.str(); }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    //!< Flag used to determine whether this is the first regime written
    bool m_FirstNonEmptyRegime;

    //!< Reference to code stream that will be used to build
    //!< entire GeNN code string e.g. a block of sim code
    std::ostringstream m_Stream;

    //! < Second internal code stream used to correctly intent the
    ::CodeStream m_CodeStream;

    //!< Internal codestream used to write regime code to
    std::ostringstream m_CurrentRegimeStream;

};

//------------------------------------------------------------------------
// PortTypeName
//------------------------------------------------------------------------
template<typename T, T InvalidPortType>
class PortTypeName
{
public:
    PortTypeName() : m_Type(InvalidPortType){}

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void set(T type, const std::string &name)
    {
        if(m_Type != InvalidPortType) {
            throw std::runtime_error("Port type and name already assigned");
        }
        else {
            m_Type = type;
            m_Name = name;
        }
    }

    std::string getName(T type) const
    {
        if(m_Type == type) {
            return m_Name;
        }
        else {
            return "";
        }
    }

    const std::string &getName() const { return m_Name; }
    T getType() const{ return m_Type; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    T m_Type;
    std::string m_Name;
};

//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------
//!< Generate model code from 'componentClass' node using specified object handlers
//!< to process various components e.g. to generate GeNN code strings
std::pair<bool, unsigned int> generateModelCode(const pugi::xml_node &componentClass,
                                                const std::map<std::string, ObjectHandler::Base*> &objectHandlerEvent,
                                                ObjectHandler::Base *objectHandlerCondition,
                                                const std::map<std::string, ObjectHandler::Base*> &objectHandlerImpulse,
                                                ObjectHandler::Base *objectHandlerTimeDerivative,
                                                std::function<void(bool, unsigned int)> regimeEndFunc);

//!< Search through code for references to named variable and replace it with text
void replaceVariableNames(std::string &code, const std::string &variableName,
                          const std::string &replaceText);

//!< Search through code for references to named variable, rename it and wrap in GeNN's $(XXXX) tags
void wrapAndReplaceVariableNames(std::string &code, const std::string &variableName,
                                 const std::string &replaceVariableName);

//!< Search through code for references to named variable and wrap in GeNN's $(XXXX) tags
void wrapVariableNames(std::string &code, const std::string &variableName);

//!< Based on the set of parameter names which are deemed to be variable,
//!< build vectors of variables and parameters to be used by GeNN model
std::tuple<NewModels::Base::StringVec, NewModels::Base::StringPairVec> findModelVariables(
    const pugi::xml_node &componentClass, const std::set<std::string> &variableParams,
    bool multipleRegimes);

void substituteModelVariables(const NewModels::Base::StringVec &paramNames,
                              const NewModels::Base::StringPairVec &vars,
                              const NewModels::Base::DerivedParamVec &derivedParams,
                              const std::vector<std::string*> &codeStrings);

// Read aliases into map
void readAliases(const pugi::xml_node &componentClass, std::map<std::string, std::string> &aliases);

// Expand out any aliases within code string
void expandAliases(std::string &code, const std::map<std::string, std::string> &aliases);

//! Return code to implement send port - will either return a variable directly or will expand an alias
std::string getSendPortCode(const std::map<std::string, std::string> &aliases,
                            const NewModels::Base::StringPairVec &vars,
                            const std::string &sendPortName);

}   // namespace SpineMLGenerator