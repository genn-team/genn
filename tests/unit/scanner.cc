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
    const auto tokens = Scanner::scanSource("1234 4294967295U -2345 -2147483647", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    ASSERT_EQ(tokens.size(), 7);
    ASSERT_EQ(tokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[2].type, Token::Type::MINUS);
    ASSERT_EQ(tokens[3].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[4].type, Token::Type::MINUS);
    ASSERT_EQ(tokens[5].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[6].type, Token::Type::END_OF_FILE);

    ASSERT_EQ(std::get<int32_t>(tokens[0].literalValue), 1234);
    ASSERT_EQ(std::get<uint32_t>(tokens[1].literalValue), 4294967295U);
    ASSERT_EQ(std::get<int32_t>(tokens[3].literalValue), 2345);
    ASSERT_EQ(std::get<int32_t>(tokens[5].literalValue), 2147483647);
}
//--------------------------------------------------------------------------
TEST(Scanner, HexInt)
{
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("0x1234 0xFFFFFFFFU -0x1234 -0x7FFFFFFF", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    ASSERT_EQ(tokens.size(), 7);
    ASSERT_EQ(tokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[2].type, Token::Type::MINUS);
    ASSERT_EQ(tokens[3].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[4].type, Token::Type::MINUS);
    ASSERT_EQ(tokens[5].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[6].type, Token::Type::END_OF_FILE);

    ASSERT_EQ(std::get<int32_t>(tokens[0].literalValue), 0x1234);
    ASSERT_EQ(std::get<uint32_t>(tokens[1].literalValue), 0xFFFFFFFFU);
    ASSERT_EQ(std::get<int32_t>(tokens[3].literalValue), 0x1234);
    ASSERT_EQ(std::get<int32_t>(tokens[5].literalValue), 0x7FFFFFFF);
}
//--------------------------------------------------------------------------
TEST(Scanner, DecimalFloat)
{
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource("1.0 0.2 100.0f 0.2f -12.0 -0.0004f", errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    ASSERT_EQ(tokens.size(), 9);
    ASSERT_EQ(tokens[0].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[1].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[2].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[3].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[4].type, Token::Type::MINUS);
    ASSERT_EQ(tokens[5].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[6].type, Token::Type::MINUS);
    ASSERT_EQ(tokens[7].type, Token::Type::NUMBER);
    ASSERT_EQ(tokens[8].type, Token::Type::END_OF_FILE);

    ASSERT_EQ(std::get<double>(tokens[0].literalValue), 1.0);
    ASSERT_EQ(std::get<double>(tokens[1].literalValue), 0.2);
    ASSERT_EQ(std::get<float>(tokens[2].literalValue), 100.0f);
    ASSERT_EQ(std::get<float>(tokens[3].literalValue), 0.2f);
    ASSERT_EQ(std::get<double>(tokens[5].literalValue), 12.0);
    ASSERT_EQ(std::get<float>(tokens[7].literalValue), 0.0004f);
}