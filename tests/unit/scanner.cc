// Google test includes
#include "gtest/gtest.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/scanner.h"

using namespace GeNN::Transpiler;


class TestErrorHandler : public ErrorHandler
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

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(Scanner, DecimalInt)
{
    TestErrorHandler errorHandler;
    const auto positiveTokens = Scanner::scanSource("1234 4294967295U", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
    ASSERT_EQ(positiveTokens.size(), 3);
    ASSERT_EQ(positiveTokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(positiveTokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(std::get<int32_t>(positiveTokens[0].literalValue), 1234);
    ASSERT_EQ(std::get<uint32_t>(positiveTokens[1].literalValue), 4294967295U);

    //const auto negativeTokens = Scanner::scanSource("-1234 -2147483648", errorHandler);
}
//--------------------------------------------------------------------------
TEST(Scanner, HexInt)
{
    TestErrorHandler errorHandler;
    const auto positiveTokens = Scanner::scanSource("0x1234 0xFFFFFFFFU", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
    ASSERT_EQ(positiveTokens.size(), 3);
    ASSERT_EQ(positiveTokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(positiveTokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(std::get<int32_t>(positiveTokens[0].literalValue), 0x1234);
    ASSERT_EQ(std::get<uint32_t>(positiveTokens[1].literalValue), 0xFFFFFFFFU);
}
//--------------------------------------------------------------------------
TEST(Scanner, DecimalFloat)
{
    TestErrorHandler errorHandler;
    const auto positiveTokens = Scanner::scanSource("1.0 0.2 100.0f 0.2f", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
    ASSERT_EQ(positiveTokens.size(), 5);
    ASSERT_EQ(positiveTokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(positiveTokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(positiveTokens[2].type, Token::Type::NUMBER);
    ASSERT_EQ(positiveTokens[3].type, Token::Type::NUMBER);
    ASSERT_EQ(std::get<double>(positiveTokens[0].literalValue), 1.0);
    ASSERT_EQ(std::get<double>(positiveTokens[1].literalValue), 0.2);
    ASSERT_EQ(std::get<float>(positiveTokens[2].literalValue), 100.0f);
    ASSERT_EQ(std::get<float>(positiveTokens[3].literalValue), 0.2f);
}