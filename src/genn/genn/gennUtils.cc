#include "gennUtils.h"

// Standard C++ includes
#include <algorithm>

// Standard C includes
#include <cctype>

// GeNN includes
#include "models.h"

namespace
{
const std::unordered_set<std::string> randomFuncs{
    "gennrand_uniform"
    "gennrand_normal",
    "gennrand_exponential",
    "gennrand_log_normal",
    "gennrand_gamma",
    "gennrand_binomial"};
}

//--------------------------------------------------------------------------
// GeNN::Utils
//--------------------------------------------------------------------------
namespace GeNN::Utils
{
bool isIdentifierReferenced(const std::string &identifierName, const std::vector<Transpiler::Token> &tokens)
{
    // Return true if any identifier's lexems match identifier name
    return std::any_of(tokens.cbegin(), tokens.cend(), 
                       [&identifierName](const auto &t)
                       { 
                           return (t.type == Transpiler::Token::Type::IDENTIFIER && t.lexeme == identifierName); 
                       });
            
}
//--------------------------------------------------------------------------
bool isRNGRequired(const std::vector<Transpiler::Token> &tokens)
{
    // Return true if any identifier's lexems are in set of random functions
    return std::any_of(tokens.cbegin(), tokens.cend(), 
                       [](const auto &t)
                       { 
                           return (t.type == Transpiler::Token::Type::IDENTIFIER && randomFuncs.find(t.lexeme) != randomFuncs.cend()); 
                       });

}
//--------------------------------------------------------------------------
bool isRNGRequired(const std::unordered_map<std::string, std::vector<Transpiler::Token>> &varInitialisers)
{
    // Return true if any of these variable initialisers require an RNG
    return std::any_of(varInitialisers.cbegin(), varInitialisers.cend(),
                       [](const auto &varInit) { return isRNGRequired(varInit.second); });
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
void validateParamNames(const std::vector<std::string> &paramNames)
{
    for(const std::string &p : paramNames) {
        validateVarName(p, "Parameter");
    }
}
}   // namespace GeNN::utils
