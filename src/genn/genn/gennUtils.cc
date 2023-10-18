#include "gennUtils.h"

// Standard C++ includes
#include <algorithm>
#include <regex>

// Standard C includes
#include <cctype>

// Platform includes
#ifdef _WIN32
#include <intrin.h>
#endif

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/scanner.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
const std::unordered_set<std::string> randomFuncs{
    "gennrand_uniform",
    "gennrand_normal",
    "gennrand_exponential",
    "gennrand_log_normal",
    "gennrand_gamma",
    "gennrand_binomial"};

std::string upgradeCodeString(const std::string &codeString)
{
    // Build vector of regular expressions to replace old style function calls
    //  **TODO** build from set of random functions
    const std::vector<std::pair<std::regex, std::string>> functionReplacements{
        {std::regex(R"(\$\(gennrand_uniform\))"), "gennrand_uniform()"},
        {std::regex(R"(\$\(gennrand_normal\))"), "gennrand_normal()"},
        {std::regex(R"(\$\(gennrand_exponential\))"), "gennrand_exponential()"},
        {std::regex(R"(\$\(gennrand_log_normal,(.*)\))"), "gennrand_log_normal($1)"},
        {std::regex(R"(\$\(gennrand_gamma,(.*)\))"), "gennrand_gamma($1)"},
        {std::regex(R"(\$\(gennrand_binomial,(.*)\))"), "gennrand_binomial($1)"},
        {std::regex(R"(\$\(addToPre,(.*)\))"), "addToPre($1)"},
        {std::regex(R"(\$\(addToInSyn,(.*)\))"), "addToPost($1)"},
        {std::regex(R"(\$\(addToInSynDelay,(.*),(.*)\))"), "addToPostDelay($1,$2)"},
        {std::regex(R"(\$\(addSynapse,(.*)\))"), "addSynapse($1)"},
        {std::regex(R"(\$\(endRow\))"), "endRow()"},
        {std::regex(R"(\$\(endCol\))"), "endCol()"}};

    // Apply sustitutions to upgraded code string
    std::string upgradedCodeString = codeString;
    for(const auto &f : functionReplacements) {
        upgradedCodeString = std::regex_replace(upgradedCodeString, f.first, f.second);
    }
    
    // **TODO** snake-case -> camel case known built in variables e.g id_pre -> idPre

    // Replace old style $(XX) variables with plain XX
    // **NOTE** this is done after functions as single-parameter function calls and variables were indistinguishable with old syntax
    const std::regex variable(R"(\$\(([_a-zA-Z][_a-zA-Z0-9]*)\))");
    upgradedCodeString = std::regex_replace(upgradedCodeString, variable, "$1");
    return upgradedCodeString;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::Utils
//--------------------------------------------------------------------------
namespace GeNN::Utils
{
std::vector<Transpiler::Token> scanCode(const std::string &code, const std::string &errorContext)
{
    using namespace Transpiler;

    // Upgrade code string
    const std::string upgradedCode = upgradeCodeString(code);

    // Scan code string and return tokens
    Transpiler::ErrorHandler errorHandler(errorContext);
    const auto tokens = Transpiler::Scanner::scanSource(upgradedCode, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Error scanning " + errorContext);
    }
    return tokens;
}
//--------------------------------------------------------------------------
Type::ResolvedType parseNumericType(const std::string &type, const Type::TypeContext &typeContext)
{
    using namespace Transpiler;

    // Scan type
    SingleLineErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(type, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Error scanning numeric type '" + std::string{type} + "'");
    }

    // Parse type numeric type
    const auto resolvedType = Parser::parseNumericType(tokens, typeContext, errorHandler);

    // If an error was encountered while scanning or parsing, throw exception
    if (errorHandler.hasError()) {
        throw std::runtime_error("Error parsing numeric type '" + std::string{type} + "'");
    }

    return resolvedType;
}
//--------------------------------------------------------------------------
bool areTokensEmpty(const std::vector<Transpiler::Token> &tokens)
{
    // For easy parsing, there should always be at least one token
    assert(tokens.size() >= 1);

    // If there's only one token, assert it is actually an EOF and return true
    if(tokens.size() == 1) {
        assert(tokens.front().type == Transpiler::Token::Type::END_OF_FILE);
        return true;
    }
    // Otherwise, return false
    else {
        return false;
    }
}
//--------------------------------------------------------------------------
bool isIdentifierReferenced(const std::string &identifierName, const std::vector<Transpiler::Token> &tokens)
{
    assert(!tokens.empty());

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
    assert(!tokens.empty());

    // Return true if any identifier's lexems are in set of random functions
    return std::any_of(tokens.cbegin(), tokens.cend(), 
                       [](const auto &t)
                       { 
                           return (t.type == Transpiler::Token::Type::IDENTIFIER && randomFuncs.find(t.lexeme) != randomFuncs.cend()); 
                       });

}
//--------------------------------------------------------------------------
bool isRNGRequired(const std::unordered_map<std::string, InitVarSnippet::Init> &varInitialisers)
{
    // Return true if any of these variable initialisers require an RNG
    return std::any_of(varInitialisers.cbegin(), varInitialisers.cend(),
                       [](const auto &varInit) { return isRNGRequired(varInit.second.getCodeTokens()); });
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
//--------------------------------------------------------------------------
std::string handleLegacyEGPType(const std::string &type)
{
    // If type string ends in *
    if(!type.empty() && type.back() == '*') {
        return type.substr(0, type.length() - 1);
    }
    // Otherwise, throw exception
    else {
        throw std::runtime_error("GeNN no longer supports non-array extra global parameters. "
                                 "Dynamic parameters provide the same functionality");
    }
}
//--------------------------------------------------------------------------
int clz(unsigned int value)
{
#ifdef _WIN32
    unsigned long leadingZero = 0;
    if(_BitScanReverse(&leadingZero, value)) {
        return 31 - leadingZero;
    }
    else {
        return 32;
    }
#else
    return __builtin_clz(value);
#endif
}
}   // namespace GeNN::utils
