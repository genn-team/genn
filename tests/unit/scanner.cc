// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "gennUtils.h"
#include "type.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
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
TEST(Scanner, DecimalInt)
{
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("1234 4294967295U -2345 -2147483647", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    ASSERT_EQ(tokens.size(), 7);
    ASSERT_EQ(tokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[0].numberType, Type::Int32);

    ASSERT_EQ(tokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[1].numberType, Type::Uint32);
    
    ASSERT_EQ(tokens[2].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[3].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[3].numberType, Type::Int32);
    
    ASSERT_EQ(tokens[4].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[5].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[5].numberType, Type::Int32);
    
    ASSERT_EQ(tokens[6].type, Token::Type::END_OF_FILE);

    ASSERT_EQ(tokens[0].lexeme, "1234");
    ASSERT_EQ(tokens[1].lexeme, "4294967295");
    ASSERT_EQ(tokens[3].lexeme, "2345");
    ASSERT_EQ(tokens[5].lexeme, "2147483647");
}
//--------------------------------------------------------------------------
TEST(Scanner, HexInt)
{
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("0x1234 0xFFFFFFFFU -0x1234 -0x7FFFFFFF", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    ASSERT_EQ(tokens.size(), 7);
    ASSERT_EQ(tokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[0].numberType, Type::Int32);

    ASSERT_EQ(tokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[1].numberType, Type::Uint32);
    
    ASSERT_EQ(tokens[2].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[3].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[3].numberType, Type::Int32);
    
    ASSERT_EQ(tokens[4].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[5].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[5].numberType, Type::Int32);
    
    ASSERT_EQ(tokens[6].type, Token::Type::END_OF_FILE);

    ASSERT_EQ(tokens[0].lexeme, "0x1234");
    ASSERT_EQ(tokens[1].lexeme, "0xFFFFFFFF");
    ASSERT_EQ(tokens[3].lexeme, "0x1234");
    ASSERT_EQ(tokens[5].lexeme, "0x7FFFFFFF");
}
//--------------------------------------------------------------------------
TEST(Scanner, DecimalFloat)
{
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("1.0 2. 0.2 100.0f 10.f 0.2f -12.0d -0.0004f 1e-4 10.0e4f -1.E-5d", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    ASSERT_EQ(tokens.size(), 15);
    ASSERT_EQ(tokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[0].numberType, std::nullopt);

    ASSERT_EQ(tokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[1].numberType, std::nullopt);
    
    ASSERT_EQ(tokens[2].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[2].numberType, std::nullopt);
    
    ASSERT_EQ(tokens[3].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[3].numberType, Type::Float);
    
    ASSERT_EQ(tokens[4].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[4].numberType, Type::Float);
    
    ASSERT_EQ(tokens[5].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[5].numberType, Type::Float);
    
    ASSERT_EQ(tokens[6].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[7].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[7].numberType, Type::Double);
    
    ASSERT_EQ(tokens[8].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[9].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[9].numberType, Type::Float);
    
    ASSERT_EQ(tokens[10].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[10].numberType, std::nullopt);
    
    ASSERT_EQ(tokens[11].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[11].numberType, Type::Float);
    
    ASSERT_EQ(tokens[12].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[13].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[13].numberType, Type::Double);
    
    ASSERT_EQ(tokens[14].type, Token::Type::END_OF_FILE);

    ASSERT_EQ(tokens[0].lexeme, "1.0");
    ASSERT_EQ(tokens[1].lexeme, "2.");
    ASSERT_EQ(tokens[2].lexeme, "0.2");
    ASSERT_EQ(tokens[3].lexeme, "100.0");
    ASSERT_EQ(tokens[4].lexeme, "10.");
    ASSERT_EQ(tokens[5].lexeme, "0.2");
    ASSERT_EQ(tokens[7].lexeme, "12.0");
    ASSERT_EQ(tokens[9].lexeme, "0.0004");
    ASSERT_EQ(tokens[10].lexeme, "1e-4");
    ASSERT_EQ(tokens[11].lexeme, "10.0e4");
    ASSERT_EQ(tokens[13].lexeme, "1.E-5");
}
//--------------------------------------------------------------------------
TEST(Scanner, DecimalFixedPoint)
{
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("1.0hk 2.hk 0.2hr 100.0hk 10.h11 0.2h11 -12.0h10 -0.0004hr 1e-4hr 10.0e4h1 -1.E-5hr", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    ASSERT_EQ(tokens.size(), 15);
    ASSERT_EQ(tokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[0].numberType, Type::S8_7);

    ASSERT_EQ(tokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[1].numberType, Type::S8_7);
    
    ASSERT_EQ(tokens[2].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[2].numberType, Type::S0_15);
    
    ASSERT_EQ(tokens[3].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[3].numberType, Type::S8_7);
    
    ASSERT_EQ(tokens[4].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[4].numberType, Type::S4_11);
    
    ASSERT_EQ(tokens[5].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[5].numberType, Type::S4_11);
    
    ASSERT_EQ(tokens[6].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[7].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[7].numberType, Type::S5_10);
    
    ASSERT_EQ(tokens[8].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[9].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[9].numberType, Type::S0_15);
    
    ASSERT_EQ(tokens[10].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[10].numberType, Type::S0_15);
    
    ASSERT_EQ(tokens[11].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[11].numberType, Type::S14_1);
    
    ASSERT_EQ(tokens[12].type, Token::Type::MINUS);
    
    ASSERT_EQ(tokens[13].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[13].numberType, Type::S0_15);
    
    ASSERT_EQ(tokens[14].type, Token::Type::END_OF_FILE);

    ASSERT_EQ(tokens[0].lexeme, "1.0");
    ASSERT_EQ(tokens[1].lexeme, "2.");
    ASSERT_EQ(tokens[2].lexeme, "0.2");
    ASSERT_EQ(tokens[3].lexeme, "100.0");
    ASSERT_EQ(tokens[4].lexeme, "10.");
    ASSERT_EQ(tokens[5].lexeme, "0.2");
    ASSERT_EQ(tokens[7].lexeme, "12.0");
    ASSERT_EQ(tokens[9].lexeme, "0.0004");
    ASSERT_EQ(tokens[10].lexeme, "1e-4");
    ASSERT_EQ(tokens[11].lexeme, "10.0e4");
    ASSERT_EQ(tokens[13].lexeme, "1.E-5");
}
//--------------------------------------------------------------------------
TEST(Scanner, String)
{
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("\"hello world\" \"pre-processor\"", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0].type, Token::Type::STRING);
    ASSERT_EQ(tokens[1].type, Token::Type::STRING);
    ASSERT_EQ(tokens[2].type, Token::Type::END_OF_FILE);

    ASSERT_EQ(tokens[0].lexeme, "\"hello world\"");
    ASSERT_EQ(tokens[1].lexeme, "\"pre-processor\"");
}
//--------------------------------------------------------------------------
TEST(Scanner, IsIdentifierDelayed)
{
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("X[10] W Z Y[3] Y[0] X Z W[10]", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    // Check delayed tokens
    ASSERT_TRUE(Utils::isIdentifierDelayed("Y", tokens));

    // Check non-delayed tokens
    ASSERT_FALSE(Utils::isIdentifierDelayed("Z", tokens));

    // Check non-existent tokens aren't delayed
    ASSERT_FALSE(Utils::isIdentifierDelayed("T", tokens));

    // Check error is thrown if identifier is referenced with and without delay
    EXPECT_THROW({ Utils::isIdentifierDelayed("X", tokens); }, 
        std::runtime_error);
    EXPECT_THROW({ Utils::isIdentifierDelayed("W", tokens); }, 
        std::runtime_error);
}