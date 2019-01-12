#include "gennUtils.h"

// Standard C++ includes
#include <algorithm>

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
bool isInitRNGRequired(const std::vector<Models::VarInit> &varInitialisers)
{
    // Return true if any of these variable initialisers require an RNG
    return std::any_of(varInitialisers.cbegin(), varInitialisers.cend(),
                       [](const Models::VarInit &varInit)
                       {
                           return isRNGRequired(varInit.getSnippet()->getCode());
                       });
}
}   // namespace utils