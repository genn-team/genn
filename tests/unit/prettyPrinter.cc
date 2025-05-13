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