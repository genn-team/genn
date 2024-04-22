// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "type.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/scanner.h"

using namespace GeNN;
using namespace GeNN::Transpiler;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class TestErrorHandler : public ErrorHandlerBase
{
public:
    TestErrorHandler() : m_Error(false)
    {}

    bool hasError() const { return m_Error; }

    virtual void error(size_t line, std::string_view message) override
    {
        report(line, "", message);
    }

    virtual void error(const Token &token, std::string_view message) override
    {
        if(token.type == Token::Type::END_OF_FILE) {
            report(token.line, " at end", message);
        }
        else {
            report(token.line, " at '" + std::string{token.lexeme} + "'", message);
        }
    }

private:
    void report(size_t line, std::string_view where, std::string_view message)
    {
        std::cerr << "[line " << line << "] Error" << where << ": " << message << std::endl;
        m_Error = true;
    }

    bool m_Error;
};
}   // Anonymous namespace

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(Parser, Numeric)
{
    // Scan
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("uint32_t", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
 
    // Parse
    ASSERT_EQ(Parser::parseNumericType(tokens, {}, errorHandler), Type::Uint32);
    ASSERT_FALSE(errorHandler.hasError());
}
//--------------------------------------------------------------------------
TEST(Parser, NumericExtraTokens)
{
    // Scan
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("uint32_t*", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    // Parse
    Parser::parseNumericType(tokens, {}, errorHandler);
    ASSERT_TRUE(errorHandler.hasError());
}
//--------------------------------------------------------------------------
TEST(Parser, Expression)
{
    // Scan
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("x + y", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
 
    // Parse
    Parser::parseExpression(tokens, {}, errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
}
//--------------------------------------------------------------------------
TEST(Parser, ExpressionActuallyMultiple)
{
    // Scan
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("x = 12; y = 37;", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
 
    // Parse
    Parser::parseExpression(tokens, {}, errorHandler);
    ASSERT_TRUE(errorHandler.hasError());   
}