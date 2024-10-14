#include "code_generator/codeGenUtils.h"

// Standard C++ library
#include <regex>

// Standard C includes
#include <cstring>

// GeNN includes
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/environment.h"
#include "code_generator/groupMerged.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/prettyPrinter.h"

//----------------------------------------------------------------------------
// GeNN::CodeGenerator
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator
{
void genTypeRange(CodeStream &os, const Type::ResolvedType &type, const std::string &prefix)
{
    const auto &numeric = type.getNumeric();
    os << "#define " << prefix << "_MIN " << Type::writeNumeric(numeric.min, type) << std::endl;

    os << "#define " << prefix << "_MAX " << Type::writeNumeric(numeric.max, type) << std::endl;
}
//----------------------------------------------------------------------------
GENN_EXPORT std::string getFastDivide(const std::string &numerator, const std::string &divisorVar)
{
    return "((((uint64_t)(" + numerator + ") * $(" + divisorVar + "_a)) + $(" + divisorVar + "_b)) >> (32 + $(" + divisorVar + "_m)))";
}
//----------------------------------------------------------------------------
void prettyPrintExpression(const std::vector<Transpiler::Token> &tokens, const Type::TypeContext &typeContext, 
                                       Transpiler::TypeChecker::EnvironmentInternal &typeCheckEnv, Transpiler::PrettyPrinter::EnvironmentInternal &prettyPrintEnv,
                                       Transpiler::ErrorHandler &errorHandler)
{
    using namespace Transpiler;

    // Parse tokens as expression
    auto expression = Parser::parseExpression(tokens, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Parse error " + errorHandler.getContext());
    }

    // Resolve types
    auto resolvedTypes = TypeChecker::typeCheck(expression.get(), typeCheckEnv, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Type check error " + errorHandler.getContext());
    }

    // Pretty print
    PrettyPrinter::print(expression, prettyPrintEnv, typeContext, resolvedTypes);
}
//----------------------------------------------------------------------------
void prettyPrintExpression(const std::vector<Transpiler::Token> &tokens, const Type::TypeContext &typeContext, 
                           EnvironmentExternalBase &env, Transpiler::ErrorHandler &errorHandler)
{
    using namespace Transpiler;

    // Create top-level internal environments and pretty-print
    TypeChecker::EnvironmentInternal typeCheckEnv(env);
    PrettyPrinter::EnvironmentInternal prettyPrintEnv(env);
    prettyPrintExpression(tokens, typeContext, typeCheckEnv, prettyPrintEnv, errorHandler);
}
 //--------------------------------------------------------------------------
void prettyPrintStatements(const std::vector<Transpiler::Token> &tokens, const Type::TypeContext &typeContext,
                           Transpiler::TypeChecker::EnvironmentInternal &typeCheckEnv, Transpiler::PrettyPrinter::EnvironmentInternal &prettyPrintEnv,
                           Transpiler::ErrorHandler &errorHandler, Transpiler::TypeChecker::StatementHandler forEachSynapseTypeCheckHandler,
                           Transpiler::PrettyPrinter::StatementHandler forEachSynapsePrettyPrintHandler)
{
    using namespace Transpiler;

    // Parse tokens as block item list (function body)
    auto updateStatements = Parser::parseBlockItemList(tokens, typeContext, errorHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Parse error " + errorHandler.getContext());
    }

    // Resolve types
    auto resolvedTypes = TypeChecker::typeCheck(updateStatements, typeCheckEnv, typeContext, 
                                                errorHandler, forEachSynapseTypeCheckHandler);
    if(errorHandler.hasError()) {
        throw std::runtime_error("Type check error " + errorHandler.getContext());
    }

    // Pretty print
    PrettyPrinter::print(updateStatements, prettyPrintEnv, typeContext, 
                         resolvedTypes, forEachSynapsePrettyPrintHandler);
}
 //--------------------------------------------------------------------------
void prettyPrintStatements(const std::vector<Transpiler::Token> &tokens, const Type::TypeContext &typeContext, EnvironmentExternalBase &env, 
                           Transpiler::ErrorHandler &errorHandler, Transpiler::TypeChecker::StatementHandler forEachSynapseTypeCheckHandler,
                           Transpiler::PrettyPrinter::StatementHandler forEachSynapsePrettyPrintHandler)
{
    using namespace Transpiler;

    // Create top-level internal environments and pretty-print
    TypeChecker::EnvironmentInternal typeCheckEnv(env);
    PrettyPrinter::EnvironmentInternal prettyPrintEnv(env);
    prettyPrintStatements(tokens, typeContext, typeCheckEnv, prettyPrintEnv, errorHandler,
                          forEachSynapseTypeCheckHandler, forEachSynapsePrettyPrintHandler);
}
//--------------------------------------------------------------------------
std::string printSubs(const std::string &format, Transpiler::PrettyPrinter::EnvironmentBase &env)
{
    // Create regex iterator to iterate over $(XXX) style varibles in format string
    // **NOTE** this doesn't match function argument $(0)
    std::regex regex("\\$\\(([a-zA-Z_][\\w]*)\\)");
    std::sregex_iterator matchesBegin(format.cbegin(), format.cend(), regex);
    std::sregex_iterator matchesEnd;
    
    // If there are no matches, leave format unmodified and return
    if(matchesBegin == matchesEnd) {
        return format;
    }
    // Otherwise
    else {
        // Loop through matches to build lazy string payload
        std::string output;
        for(std::sregex_iterator m = matchesBegin;;) {
            // Copy the non-matched subsequence (m->prefix()) onto output
            std::copy(m->prefix().first, m->prefix().second, std::back_inserter(output));

            // Add environment value of $(XXX) to output
            output += env[(*m)[1]];
    
            // If there are no subsequent matches, add the remaining 
            // non-matched characters onto output and return
            if(std::next(m) == matchesEnd) {
                 // Copy the non-matched subsequence (m->prefix()) onto output
                 std::copy(m->suffix().first, m->suffix().second, std::back_inserter(output));
                 return output;
            }
            // Otherwise go onto next match
            else {
                m++;
            }
        }
    }
}
}   // namespace GeNN::CodeGenerator
