#include "gennUtils.h"

// Standard C++ includes
#include <algorithm>

// Standard C includes
#include <cctype>

// GeNN includes
#include "models.h"

namespace
{
//--------------------------------------------------------------------------
// GenericFunction
//--------------------------------------------------------------------------
//! Immutable structure for specifying the name and number of
//! arguments of a generic funcion e.g. gennrand_uniform
struct GenericFunction
{
    //! Generic name used to refer to function in user code
    const std::string genericName;

    //! Number of function arguments
    const unsigned int numArguments;
};


GenericFunction randomFuncs[] = {
    {"gennrand_uniform", 0},
    {"gennrand_normal", 0},
    {"gennrand_exponential", 0},
    {"gennrand_log_normal", 2},
    {"gennrand_gamma", 1}
};
}

//--------------------------------------------------------------------------
// Utils
//--------------------------------------------------------------------------
namespace Utils
{
bool isRNGRequired(const std::string &code)
{
    // Loop through random functions
    for(const auto &r : randomFuncs) {
        // If this function takes no arguments, return true if
        // generic function name enclosed in $() markers is found
        if(r.numArguments == 0) {
            if(code.find("$(" + r.genericName + ")") != std::string::npos) {
                return true;
            }
        }
        // Otherwise, return true if generic function name
        // prefixed by $( and suffixed with comma is found
        else if(code.find("$(" + r.genericName + ",") != std::string::npos) {
            return true;
        }
    }
    return false;

}
//--------------------------------------------------------------------------
bool isRNGRequired(const std::vector<Models::VarInit> &varInitialisers)
{
    // Return true if any of these variable initialisers require an RNG
    return std::any_of(varInitialisers.cbegin(), varInitialisers.cend(),
                       [](const Models::VarInit &varInit)
                       {
                           return isRNGRequired(varInit.getSnippet()->getCode());
                       });
}
//--------------------------------------------------------------------------
bool isTypePointer(const std::string &type)
{
    return (type.back() == '*');
}
//--------------------------------------------------------------------------
bool isTypePointerToPointer(const std::string &type)
{
    const size_t len = type.length();
    return (type[len - 1] == '*' && type[len - 2] == '*');
}
//--------------------------------------------------------------------------
bool isTypeFloatingPoint(const std::string &type)
{
    assert(!isTypePointer(type));
    return ((type == "float") || (type == "double") || (type == "half") || (type == "scalar"));
}
//--------------------------------------------------------------------------
std::string getUnderlyingType(const std::string &type)
{
    // Check that type is a pointer type
    assert(isTypePointer(type));

    // if type is actually a pointer to a pointer, return string without last 2 characters
    if(isTypePointerToPointer(type)) {
        return type.substr(0, type.length() - 2);
    }
    // Otherwise, return string without last character
    else {
        return type.substr(0, type.length() - 1);
    }
}
//--------------------------------------------------------------------------
void validateVarName(const std::string &name, const std::string &description)
{
    // Empty names aren't valid
    if(name.empty()) {
        throw std::runtime_error(description + " name invalid: cannot be empty");
    }

    // If first character's a number, name isn't valid
    if(std::isdigit(name.front())) {
        throw std::runtime_error(description + " name invalid: '" + name + "' starts with a digit");
    }
    
    // If any characters aren't underscores or alphanumeric, name isn't valud
    if(std::any_of(name.cbegin(), name.cend(),
                   [](char c) { return (c != '_') && !std::isalnum(c); }))
    {
        throw std::runtime_error(description + " name invalid: '" + name + "' contains an illegal character");
    }
}
//--------------------------------------------------------------------------
void validatePopName(const std::string &name, const std::string &description)
{
    // Empty names aren't valid
    if(name.empty()) {
        throw std::runtime_error(description + " name invalid: cannot be empty");
    }

    // If any characters aren't underscores or alphanumeric, name isn't valid
    if(std::any_of(name.cbegin(), name.cend(),
                   [](char c) { return (c != '_') && !std::isalnum(c); }))
    {
        throw std::runtime_error(description + " name invalid: '" + name + "' contains an illegal character");
    }
}
//--------------------------------------------------------------------------
void validateParamValues(const std::vector<std::string> &paramNames, const Snippet::ParamValues &paramValues, 
                         const std::string &description) 
{
    // Loop through names
    for(const auto &n : paramNames) {
        // If there is no values, give error
        if(paramValues.getValues().find(n) == paramValues.getValues().cend()) {
            throw std::runtime_error(description + " missing value for parameter: '" + n + "'");
        }
    }
}
//--------------------------------------------------------------------------
void validateParamNames(const std::vector<std::string> &paramNames)
{
    for(const std::string &p : paramNames) {
        validateVarName(p, "Parameter");
    }
}
}   // namespace utils
