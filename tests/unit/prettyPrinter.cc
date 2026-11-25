// Standard C++ includes
#include <sstream>

// Google test includes
#include "gtest/gtest.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/scanner.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/environment.h"
#include "code_generator/standardLibrary.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;


//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(PrettyPrinter, Printf)
{
    ErrorHandler errorHandler("");

    // Printf with no variadic arguments
    {
        std::ostringstream stringStream;
        CodeStream codeStream(stringStream);

        EnvironmentLibrary env(codeStream, StandardLibrary::getMathsFunctions());
        const auto tokens = Scanner::scanSource("printf(\"hello\");", errorHandler);
        prettyPrintStatements(tokens, {}, env, errorHandler, nullptr, nullptr);

        ASSERT_EQ(stringStream.str(), "printf(\"hello\");\n");
    }

    // Printf with arguments
    {
        std::ostringstream stringStream;
        CodeStream codeStream(stringStream);

        EnvironmentLibrary env(codeStream, StandardLibrary::getMathsFunctions());
        const auto tokens = Scanner::scanSource("printf(\"hello %d, %f\", 12, 15.0f);", errorHandler);
        prettyPrintStatements(tokens, {}, env, errorHandler, nullptr, nullptr);

        ASSERT_EQ(stringStream.str(), "printf(\"hello %d, %f\", 12, 15.0f);\n");
    }
 
}
//--------------------------------------------------------------------------
TEST(PrettyPrinter, ArgumentParentheses)
{
    // Library containing function and array subscript override
    // whose placeholders are surrounded by operators
    const EnvironmentLibrary::Library library = {
        {"func", {Type::ResolvedType::createFunction(Type::Int32, {Type::Int32}), "((10 - $(0)) % 5)"}},
        {"array", {Type::getArraySubscript(Type::Int32), "array[(10 - $(0)) % 5]"}}};

    const auto prettyPrint =
        [&library](const std::string &code)
        {
            ErrorHandler errorHandler("");
            std::ostringstream stringStream;
            CodeStream codeStream(stringStream);

            EnvironmentLibrary env(codeStream, library);
            const auto tokens = Scanner::scanSource(code, errorHandler);
            prettyPrintStatements(tokens, {}, env, errorHandler, nullptr, nullptr);
            return stringStream.str();
        };

    // Arguments containing operators should be parenthesised
    ASSERT_EQ(prettyPrint("func(1 + 2);"), "((10 - (1 + 2)) % 5);\n");
    ASSERT_EQ(prettyPrint("func(1 ? 2 : 3);"), "((10 - (1 ? 2 : 3)) % 5);\n");
    ASSERT_EQ(prettyPrint("func(-1);"), "((10 - (-1)) % 5);\n");
    ASSERT_EQ(prettyPrint("array[1 + 2];"), "array[(10 - (1 + 2)) % 5];\n");

    // Other arguments should be unchanged
    ASSERT_EQ(prettyPrint("func(1);"), "((10 - 1) % 5);\n");
    ASSERT_EQ(prettyPrint("func((1 + 2));"), "((10 - (1 + 2)) % 5);\n");
    ASSERT_EQ(prettyPrint("func(func(1));"), "((10 - ((10 - 1) % 5)) % 5);\n");
    ASSERT_EQ(prettyPrint("func(array[1]);"), "((10 - array[(10 - 1) % 5]) % 5);\n");
}
