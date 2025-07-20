#include "transpiler/scanner.h"

// Standard C++ includes
#include <functional>
#include <map>
#include <set>
#include <unordered_map>

// Standard C includes
#include <cassert>
#include <cctype>

// Transpiler includes
#include "transpiler/errorHandler.h"

using namespace GeNN;
using namespace GeNN::Transpiler;
using namespace GeNN::Transpiler::Scanner;

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
const std::unordered_map<std::string_view, Token::Type> keywords{
    {"const", Token::Type::TYPE_QUALIFIER},
    {"do", Token::Type::DO},
    {"else", Token::Type::ELSE},
    {"false", Token::Type::BOOLEAN},
    {"for", Token::Type::FOR},
    {"for_each_synapse", Token::Type::FOR_EACH_SYNAPSE},
    {"if", Token::Type::IF},
    {"true", Token::Type::BOOLEAN},
    {"while", Token::Type::WHILE},
    {"switch", Token::Type::SWITCH},
    {"break", Token::Type::BREAK},
    {"continue", Token::Type::CONTINUE},
    {"case", Token::Type::CASE},
    {"default", Token::Type::DEFAULT},
    {"char", Token::Type::TYPE_SPECIFIER},
    {"short", Token::Type::TYPE_SPECIFIER},
    {"int", Token::Type::TYPE_SPECIFIER},
    {"long", Token::Type::TYPE_SPECIFIER},
    {"float", Token::Type::TYPE_SPECIFIER},
    {"double", Token::Type::TYPE_SPECIFIER},
    {"signed", Token::Type::TYPE_SPECIFIER},
    {"unsigned", Token::Type::TYPE_SPECIFIER},
    {"uint8_t", Token::Type::TYPE_SPECIFIER},
    {"int8_t", Token::Type::TYPE_SPECIFIER},
    {"uint16_t", Token::Type::TYPE_SPECIFIER},
    {"int16_t", Token::Type::TYPE_SPECIFIER},
    {"uint32_t", Token::Type::TYPE_SPECIFIER},
    {"uint64_t", Token::Type::TYPE_SPECIFIER},
    {"int32_t", Token::Type::TYPE_SPECIFIER},
    {"int64_t", Token::Type::TYPE_SPECIFIER},
    {"size_t", Token::Type::TYPE_SPECIFIER},
    {"bool", Token::Type::TYPE_SPECIFIER},
    {"scalar", Token::Type::TYPE_SPECIFIER},
    {"timepoint", Token::Type::TYPE_SPECIFIER}};
//---------------------------------------------------------------------------
// ScanState
//---------------------------------------------------------------------------
//! Class encapsulated logic to navigate through source characters
class ScanState
{
public:
    ScanState(std::string_view source, ErrorHandlerBase &errorHandler)
        : m_Start(0), m_Current(0), m_Line(1), m_Source(source), m_ErrorHandler(errorHandler)
    {
    }

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    char advance() {
        m_Current++;
        return m_Source.at(m_Current - 1);
    }

    bool match(char expected)
    {
        if(isAtEnd()) {
            return false;
        }
        if(m_Source.at(m_Current) != expected) {
            return false;
        }

        m_Current++;
        return true;
    }

    void resetLexeme()
    {
        m_Start = m_Current;
    }

    char peek() const
    {
        if(isAtEnd()) {
            return '\0';
        }
        return m_Source.at(m_Current);
    }

    char peekNext() const
    {
        if((m_Current + 1) >= m_Source.length()) {
            return '\0';
        }
        else {
            return m_Source.at(m_Current + 1);
        }
    }

    std::string_view getLexeme() const
    {
        return m_Source.substr(m_Start, m_Current - m_Start);
    }

    size_t getLine() const { return m_Line; }

    bool isAtEnd() const { return m_Current >= m_Source.length(); }

    void nextLine() { m_Line++; }

    void error(std::string_view message)
    {
        m_ErrorHandler.error(getLine(), message);
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    size_t m_Start;
    size_t m_Current;
    size_t m_Line;

    std::string_view m_Source;
    ErrorHandlerBase &m_ErrorHandler;
};

bool isodigit(char c)
{
    return (c >= '0' && c <= '7');
}

//---------------------------------------------------------------------------
void emplaceToken(std::vector<Token> &tokens, Token::Type type, const ScanState &scanState)
{
    tokens.emplace_back(type, scanState.getLexeme(), scanState.getLine());
}
//---------------------------------------------------------------------------
void scanNumber(char c, ScanState &scanState, std::vector<Token> &tokens)
{
    // If this is a hexadecimal literal
    if(c == '0' && (scanState.match('x') || scanState.match('X'))) {
        // Read hexadecimal digits
        while(std::isxdigit(scanState.peek())) {
            scanState.advance();
        }

        // If a decimal point is found, give an error
        if(scanState.match('.')) {
            scanState.error("Hexadecimal floating pointer literals unsupported.");
        }

        // Read hexadecimal digits
        while(std::isxdigit(scanState.peek())) {
            scanState.advance();
        }

        // If there's a U suffix, emplace 
        if (std::toupper(scanState.peek()) == 'U') {
            emplaceToken(tokens, Token::Type::UINT32_NUMBER, scanState);
            scanState.advance();
        }
        else {
            emplaceToken(tokens, Token::Type::INT32_NUMBER, scanState);
        }
        
    }
    // Otherwise, if this is an octal integer
    else if(c == '0' && isodigit(scanState.peek())){
        scanState.error("Octal literals unsupported.");
    }
    // Otherwise, if it's decimal
    else {
        // Read digits
        while(std::isdigit(scanState.peek())) {
            scanState.advance();
        }

        // Read decimal place
        bool isFloat = scanState.match('.');

        // Read digits
        while(std::isdigit(scanState.peek())) {
            scanState.advance();
        }

        // If there's an exponent
        if(scanState.match('e') || scanState.match('E')) {
            // Must be floating point, whatever precedes
            isFloat = true;

            // Read sign
            if(scanState.peek() == '-' || scanState.peek() == '+') {
                scanState.advance();
            }

            // Read digits
            while(std::isdigit(scanState.peek())) {
                scanState.advance();
            }
        }
        
        // If it's float
        if(isFloat) {
            // If number has an f suffix, emplace FLOAT_NUMBER token
            if (std::tolower(scanState.peek()) == 'f') {
                emplaceToken(tokens, Token::Type::FLOAT_NUMBER, scanState);
                scanState.advance();
            }
            // Otherwise, if it has a d suffix, emplace DOUBLE_NUMBER token
            // **NOTE** 'd' is a GeNN extension not standard C
            else if (std::tolower(scanState.peek()) == 'd') {
                emplaceToken(tokens, Token::Type::DOUBLE_NUMBER, scanState);
                scanState.advance();
            }
            // Otherwise, emplace scalar literal whose type will be decoded later
            else {
                emplaceToken(tokens, Token::Type::SCALAR_NUMBER, scanState);
            }
        }
        // Otherwise, emplace integer token 
        else {
            // If there's a U suffix, emplace 
            if (std::toupper(scanState.peek()) == 'U') {
                emplaceToken(tokens, Token::Type::UINT32_NUMBER, scanState);
                scanState.advance();
            }
            else {
                emplaceToken(tokens, Token::Type::INT32_NUMBER, scanState);
            }
        }
    }
}
//---------------------------------------------------------------------------
void scanString(ScanState &scanState, std::vector<Token> &tokens)
{
    // Loop
    // **TODO** more complex logic here
    while(true) {
        // If we have reached end of string, give error
        if(scanState.isAtEnd()) {
            scanState.error("Unterminated string literal encountered.");
            return;
        }
        // Otherwise, if next character is closing quote, stop
        else if(scanState.peek() == '"') {
            break;
        }
        // Otherwise, advance over string
        else {
            scanState.advance();
        }
    }

    // Match closing quote and emplace string token
    scanState.match('"');
    emplaceToken(tokens, Token::Type::STRING, scanState);
}
//---------------------------------------------------------------------------
void scanIdentifier(ScanState &scanState, std::vector<Token> &tokens)
{
    // Read subsequent alphanumeric characters and underscores
    while(std::isalnum(scanState.peek()) || scanState.peek() == '_') {
        scanState.advance();
    }

    // If identifier is a keyword, add appropriate token
    const auto k = keywords.find(scanState.getLexeme());
    if(k != keywords.cend()) {
        emplaceToken(tokens, k->second, scanState);
    }
    // Otherwise, add identifier token
    else {
        emplaceToken(tokens, Token::Type::IDENTIFIER, scanState);
    }
}
//---------------------------------------------------------------------------
void scanToken(ScanState &scanState, std::vector<Token> &tokens)
{
    char c = scanState.advance();
    switch(c) {
        // Single character tokens
        case '(': emplaceToken(tokens, Token::Type::LEFT_PAREN, scanState); break;
        case ')': emplaceToken(tokens, Token::Type::RIGHT_PAREN, scanState); break;
        case '{': emplaceToken(tokens, Token::Type::LEFT_BRACE, scanState); break;
        case '}': emplaceToken(tokens, Token::Type::RIGHT_BRACE, scanState); break;
        case '[': emplaceToken(tokens, Token::Type::LEFT_SQUARE_BRACKET, scanState); break;
        case ']': emplaceToken(tokens, Token::Type::RIGHT_SQUARE_BRACKET, scanState); break;
        case ',': emplaceToken(tokens, Token::Type::COMMA, scanState); break;
        case '.': emplaceToken(tokens, Token::Type::DOT, scanState); break;
        case ':': emplaceToken(tokens, Token::Type::COLON, scanState); break;
        case ';': emplaceToken(tokens, Token::Type::SEMICOLON, scanState); break;
        case '~': emplaceToken(tokens, Token::Type::TILDA, scanState); break;
        case '?': emplaceToken(tokens, Token::Type::QUESTION, scanState); break;

        // Operators
        case '!': emplaceToken(tokens, scanState.match('=') ? Token::Type::NOT_EQUAL : Token::Type::NOT, scanState); break;
        case '=': emplaceToken(tokens, scanState.match('=') ? Token::Type::EQUAL_EQUAL : Token::Type::EQUAL, scanState); break;

        // Assignment operators
        case '*': emplaceToken(tokens, scanState.match('=') ? Token::Type::STAR_EQUAL : Token::Type::STAR, scanState); break;
        //case '/': emplaceToken(tokens, scanState.match('=') ? Token::Type::SLASH_EQUAL : Token::Type::SLASH, scanState); break;
        case '%': emplaceToken(tokens, scanState.match('=') ? Token::Type::PERCENT_EQUAL : Token::Type::PERCENT, scanState); break;
        case '^': emplaceToken(tokens, scanState.match('=') ? Token::Type::CARET_EQUAL : Token::Type::CARET, scanState); break;

        case '<': 
        {
            if(scanState.match('=')) {
                emplaceToken(tokens, Token::Type::LESS_EQUAL, scanState);
            }
            else if(scanState.match('<')) {
                if(scanState.match('=')) {
                    emplaceToken(tokens, Token::Type::SHIFT_LEFT_EQUAL, scanState);
                }
                else {
                    emplaceToken(tokens, Token::Type::SHIFT_LEFT, scanState);
                }
            }
            else {
                emplaceToken(tokens, Token::Type::LESS, scanState);
            }
            break;
        }

        case '>': 
        {
            if(scanState.match('=')) {
                emplaceToken(tokens, Token::Type::GREATER_EQUAL, scanState);
            }
            else if(scanState.match('<')) {
                if(scanState.match('=')) {
                    emplaceToken(tokens, Token::Type::SHIFT_RIGHT_EQUAL, scanState);
                }
                else {
                    emplaceToken(tokens, Token::Type::SHIFT_RIGHT, scanState);
                }
            }
            else {
                emplaceToken(tokens, Token::Type::GREATER, scanState);
            }
            break;
        }

        case '+':
        {
            if(scanState.match('=')) {
                emplaceToken(tokens, Token::Type::PLUS_EQUAL, scanState);
            }
            else if(scanState.match('+')) {
                emplaceToken(tokens, Token::Type::PLUS_PLUS, scanState);
            }
            else {
                emplaceToken(tokens, Token::Type::PLUS, scanState);
            }
            break;
        }

        case '-':
        {
            if(scanState.match('=')) {
                emplaceToken(tokens, Token::Type::MINUS_EQUAL, scanState);
            }
            else if(scanState.match('-')) {
                emplaceToken(tokens, Token::Type::MINUS_MINUS, scanState);
            }
            else {
                emplaceToken(tokens, Token::Type::MINUS, scanState);
            }
            break;
        }

        case '&': 
        {
            if(scanState.match('=')) {
                emplaceToken(tokens, Token::Type::AMPERSAND_EQUAL, scanState);
            }
            else if(scanState.match('&')) {
                emplaceToken(tokens, Token::Type::AMPERSAND_AMPERSAND, scanState);
            }
            else {
                emplaceToken(tokens, Token::Type::AMPERSAND, scanState);
            }
            break;
        }

        case '|': 
        {
            if(scanState.match('=')) {
                emplaceToken(tokens, Token::Type::PIPE_EQUAL, scanState);
            }
            else if(scanState.match('|')) {
                emplaceToken(tokens, Token::Type::PIPE_PIPE, scanState);
            }
            else {
                emplaceToken(tokens, Token::Type::PIPE, scanState);
            }
            break;
        }

        case '/':
        {
            // Line comment
            if(scanState.match('/')) {
                while(scanState.peek() != '\n' && !scanState.isAtEnd()) {
                    scanState.advance();
                }
            }
            else {
                emplaceToken(tokens, Token::Type::SLASH, scanState);
            }
            break;
        }

        // String
        case '"': scanString(scanState, tokens); break;

        // Whitespace
        case ' ':
        case '\r':
        case '\t':
            break;

        // New line
        case '\n': scanState.nextLine(); break;

        default:
        {
            // If we have a digit or a period, scan number
            if(std::isdigit(c) || c == '.') {
                scanNumber(c, scanState, tokens);
            }
            // Otherwise, scan identifier
            else if(std::isalpha(c)) {
                scanIdentifier(scanState, tokens);
            }
            else {
                scanState.error("Unexpected character '" + std::string{c} + "'.");
            }
        }
    }
}
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Scanner
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Scanner
{
std::vector<Token> scanSource(const std::string_view &source, ErrorHandlerBase &errorHandler)
{
    std::vector<Token> tokens;

    ScanState scanState(source, errorHandler);

    // Scan tokens
    while(!scanState.isAtEnd()) {
        scanState.resetLexeme();
        scanToken(scanState, tokens);
    }

    emplaceToken(tokens, Token::Type::END_OF_FILE, scanState);
    return tokens;
}
}
