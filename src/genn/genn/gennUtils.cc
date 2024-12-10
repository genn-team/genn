#include "gennUtils.h"

// Standard C++ includes
#include <algorithm>
#include <optional>

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
    "gennrand",
    "gennrand_uniform",
    "gennrand_normal",
    "gennrand_exponential",
    "gennrand_log_normal",
    "gennrand_gamma",
    "gennrand_binomial"};
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::Utils
//--------------------------------------------------------------------------
namespace GeNN::Utils
{
std::vector<Transpiler::Token> scanCode(const std::string &code, const std::string &errorContext)
{
    using namespace Transpiler;

    // Scan code string and return tokens
    Transpiler::ErrorHandler errorHandler(errorContext);
    const auto tokens = Transpiler::Scanner::scanSource(code, errorHandler);
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
bool isIdentifierDelayed(const std::string &identifierName, const std::vector<Transpiler::Token> &tokens)
{
    // Loop through tokens
    std::optional<bool> delayed;
    for(auto t = tokens.cbegin(); t != tokens.cend(); t++) {
        // If token is an identifier with correct name
        if(t->type == Transpiler::Token::Type::IDENTIFIER && t->lexeme == identifierName) {
            // If token isn't last in sequence and it's followed by a left square bracket
            const auto tNext = std::next(t);
            if(tNext != tokens.cend() && tNext->type == Transpiler::Token::Type::LEFT_SQUARE_BRACKET) {
                // If identifier hasn't been encountered before, mark as delayed
                if(!delayed.has_value()) {
                    delayed = true;
                }
                // Otherwise, if this identifier was previous encountered without delay, give error
                else if(!delayed.value()) {
                    throw std::runtime_error("Identifier '" + identifierName + "' referenced both with and without delay");
                }
            }
            // Otherwise
            else {
                // If identifier hasn't been encountered before, mark as non-delayed
                if(!delayed.has_value()) {
                    delayed = false;
                }
                // Otherwise, if this identifier was previous encountered with delay, give error
                else if(delayed.value()) {
                    throw std::runtime_error("Identifier '" + identifierName + "' referenced both with and without delay");
                }
            }
        }
    }

    // Return true if identifier encountered delayed
    return delayed.value_or(false);
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
bool isRNGRequired(const std::map<std::string, InitVarSnippet::Init> &varInitialisers)
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
//--------------------------------------------------------------------------
int ctz(unsigned int value)
{
#ifdef _WIN32
    unsigned long trailingZero = 0;
    if(_BitScanForward(&trailingZero, value)) {
        return trailingZero;
    }
    else {
        return 32;
    }
#else
    return __builtin_ctz(value);
#endif
}
//--------------------------------------------------------------------------
int popCount(unsigned int value)
{
#ifdef _WIN32
    return __popcnt(value);
#else
    return __builtin_popcount(value);
#endif
}
}   // namespace GeNN::utils
