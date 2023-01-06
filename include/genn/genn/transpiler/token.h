#pragma once

// Standard C++ includes
#include <string_view>
#include <variant>

// Standard C includes
#include <cstdint>

//---------------------------------------------------------------------------
// MiniParse::Token
//---------------------------------------------------------------------------
namespace MiniParse
{
struct Token
{
    typedef std::variant<std::monostate, bool, float, double, uint32_t, int32_t/*, uint64_t, int64_t*/> LiteralValue;

    enum class Type
    {
        // Single-character tokens
        LEFT_PAREN, RIGHT_PAREN, LEFT_BRACE, RIGHT_BRACE, LEFT_SQUARE_BRACKET, RIGHT_SQUARE_BRACKET,
        COMMA, PIPE, CARET, DOT, MINUS, PERCENT, PLUS, COLON, SEMICOLON, SLASH, STAR, TILDA, AMPERSAND, QUESTION,

        // One or two character tokens
        NOT, NOT_EQUAL,
        EQUAL_EQUAL,
        GREATER, GREATER_EQUAL,
        LESS, LESS_EQUAL,
        EQUAL, STAR_EQUAL, SLASH_EQUAL, PERCENT_EQUAL, PLUS_EQUAL,
        MINUS_EQUAL, AMPERSAND_EQUAL, CARET_EQUAL, PIPE_EQUAL,
        PIPE_PIPE, AMPERSAND_AMPERSAND, PLUS_PLUS, MINUS_MINUS,
        SHIFT_LEFT, SHIFT_RIGHT,

        // Three character tokens
        SHIFT_LEFT_EQUAL, SHIFT_RIGHT_EQUAL,

        // Literals   
        IDENTIFIER, NUMBER,

        // Types
        TYPE_SPECIFIER,
        TYPE_QUALIFIER,

        // Keywords
        DO, ELSE, FALSE, FOR, IF, TRUE, WHILE, PRINT, SWITCH, CONTINUE, BREAK, CASE, DEFAULT,

        END_OF_FILE,
    };

    Token(Type type, std::string_view lexeme, size_t line, LiteralValue literalValue = LiteralValue())
        : type(type), lexeme(lexeme), line(line), literalValue(literalValue)
    {
    }

    const Type type;
    const std::string_view lexeme;
    const size_t line;
    const LiteralValue literalValue;
};

}
