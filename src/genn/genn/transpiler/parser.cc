#include "transpiler/parser.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>

// Standard C includes
#include <cassert>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/errorHandler.h"

using namespace GeNN;
using namespace GeNN::Transpiler;
using namespace GeNN::Transpiler::Parser;

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
const std::map<std::multiset<std::string>, Type::ResolvedType> numericTypeSpecifiers{
    {{"bool"}, Type::Bool},

    {{"char"}, Type::Int8},
    {{"int8_t"}, Type::Int8},
    
    {{"unsigned", "char"}, Type::Uint8},
    {{"uint8_t"}, Type::Uint8},

    {{"short"}, Type::Int16},
    {{"short", "int"}, Type::Int16},
    {{"signed", "short"}, Type::Int16},
    {{"signed", "short", "int"}, Type::Int16},
    {{"int16_t"}, Type::Int16},
    
    {{"unsigned", "short"}, Type::Uint16},
    {{"unsigned", "short", "int"}, Type::Uint16},
    {{"uint16_t"}, Type::Uint16},

    {{"int"}, Type::Int32},
    {{"signed"}, Type::Int32},
    {{"signed", "int"}, Type::Int32},
    {{"int32_t"}, Type::Int32},

    {{"unsigned"}, Type::Uint32},
    {{"unsigned", "int"}, Type::Uint32},
    {{"uint32_t"}, Type::Uint32},

    // **NOTE** GeNN uses LP64 data model where longs are 64-bit (unlike Windows)
    {{"long"}, Type::Int64},
    {{"long", "int"}, Type::Int64},
    {{"signed", "long"}, Type::Int64},
    {{"signed", "long", "int"}, Type::Int64},
    {{"long", "long"}, Type::Int64},
    {{"long", "long", "int"}, Type::Int64},
    {{"signed", "long", "long"}, Type::Int64},
    {{"signed", "long", "long", "int"}, Type::Int64},
    {{"int64_t"}, Type::Int64},

    // **NOTE** GeNN uses LP64 data model where longs are 64-bit (unlike Windows)
    {{"unsigned", "long"}, Type::Uint64},
    {{"unsigned", "long", "int"}, Type::Uint64},
    {{"unsigned", "long"}, Type::Uint64},
    {{"unsigned", "long", "long", "int"}, Type::Uint64},
    {{"uint64_t"}, Type::Uint64},
    {{"size_t"}, Type::Uint64},

    {{"float"}, Type::Float},
    {{"double"}, Type::Double},
    
    {{"half"}, Type::Half},
    {{"bfloat16"}, Type::Bfloat16}};

//---------------------------------------------------------------------------
// ParserState
//---------------------------------------------------------------------------
//! Class encapsulated logic to navigate through tokens
class ParserState
{
public:
    ParserState(const std::vector<Token> &tokens, const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
        : m_Current(0), m_Tokens(tokens), m_Context(context), m_ErrorHandler(errorHandler)
    {}

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    bool match(Token::Type t)
    {
        if(check(t)) {
            advance();
            return true;
        }
        else {
            return false;
        }
    }

    bool match(std::initializer_list<Token::Type> types) 
    {
        // Loop through types
        for(auto t : types) {
            if(match(t)) {
                return true;
            }
        }
        return false;
    }

    Token advance()
    {
        if(!isAtEnd()) {
            m_Current++;
        }

        return previous();
    }

    Token rewind()
    {
        if(m_Current > 0) {
            m_Current--;
        }

        return peek();
    }

    Token peek() const
    {
        return m_Tokens.at(m_Current);
    }

    Token previous() const
    {
        assert(m_Current > 0);
        return m_Tokens.at(m_Current - 1);
    }

    void error(std::string_view message) const
    {
        m_ErrorHandler.error(peek(), message);
    }

    void error(Token token, std::string_view message) const
    {
        m_ErrorHandler.error(token, message);
    }

    Token consume(Token::Type type, std::string_view message) 
    {
        if(check(type)) {
            return advance();
        }

        error(message);
        throw ParseError();
     }

    bool check(Token::Type type) const
    {
        if(isAtEnd()) {
            return false;
        }
        else {
            return (peek().type == type);
        }
    }

    bool isAtEnd() const { return (peek().type == Token::Type::END_OF_FILE); }
    
    const Type::TypeContext &getContext() const{ return m_Context; }
    
private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    size_t m_Current;

    const std::vector<Token> &m_Tokens;
    const Type::TypeContext &m_Context;
    ErrorHandlerBase &m_ErrorHandler;
};

// **THINK** could leave unresolved
Type::ResolvedType getNumericType(const std::multiset<std::string> &typeSpecifiers, const Type::TypeContext &context)
{
    // If type is numeric, return 
    const auto type = numericTypeSpecifiers.find(typeSpecifiers);
    if (type != numericTypeSpecifiers.cend()) {
        return type->second;
    }
    else {
        // **YUCK** use sets everywhere
        if (typeSpecifiers.size() == 1) {
            const auto contextType = context.find(*typeSpecifiers.begin());
            if (contextType != context.cend()) {
                return contextType->second;
            }
        }

        // Generate string representation of type specifier and give error
        std::ostringstream typeSpecifiersString;
        std::copy(typeSpecifiers.cbegin(), typeSpecifiers.cend(),
                  std::ostream_iterator<std::string>(typeSpecifiersString, " "));
        throw std::runtime_error("Unknown numeric type specifier '" + typeSpecifiersString.str() + "'");
    }
}

void synchronise(ParserState &parserState)
{
    parserState.advance();
    while(!parserState.isAtEnd()) {
        if(parserState.previous().type == Token::Type::SEMICOLON) {
            return;
        }
    
        const auto nextTokenType = parserState.peek().type;
        if(nextTokenType == Token::Type::FOR
           || nextTokenType == Token::Type::IF
           || nextTokenType == Token::Type::WHILE
           || nextTokenType == Token::Type::TYPE_SPECIFIER)
        {
            return;
        }

        parserState.advance();
    }
}

// Forward declarations
Expression::ExpressionPtr parseCast(ParserState &parserState);
Expression::ExpressionPtr parseAssignment(ParserState &parserState);
Expression::ExpressionPtr parseExpression(ParserState &parserState);
Statement::StatementPtr parseBlockItem(ParserState &parserState);
Statement::StatementPtr parseDeclaration(ParserState &parserState);
Statement::StatementPtr parseStatement(ParserState &parserState);

// Helper to parse binary expressions
// **THINK I think this COULD be variadic but not clear if that's a good idea or not
template<typename N>
Expression::ExpressionPtr parseBinary(ParserState &parserState, N nonTerminal, std::initializer_list<Token::Type> types)
{
    auto expression = nonTerminal(parserState);
    while(parserState.match(types)) {
        Token op = parserState.previous();
        expression = std::make_unique<Expression::Binary>(std::move(expression), op, nonTerminal(parserState));
    }

    return expression;
}

GeNN::Type::ResolvedType parseDeclarationSpecifiers(ParserState &parserState)
{
    using namespace GeNN::Type;
    
    std::multiset<std::string> typeSpecifiers;
    std::set<std::string> typeQualifiers;
    std::vector<std::set<std::string>> pointerTypeQualifiers;
    
    do {
        // If token is a star, add new set of pointer type qualifiers
        if(parserState.previous().type == Token::Type::STAR) {
            pointerTypeQualifiers.emplace_back();
        }
        // Otherwise, if type is a qualifier
        else if(parserState.previous().type == Token::Type::TYPE_QUALIFIER) {
            // Add qualifier lexeme to correct list
            std::set<std::string> &qualifiers = pointerTypeQualifiers.empty() ? typeQualifiers : pointerTypeQualifiers.back();
            if(!qualifiers.insert(parserState.previous().lexeme).second) {
                parserState.error(parserState.previous(), "duplicate type qualifier");
            }
        }
        else if(parserState.previous().type == Token::Type::TYPE_SPECIFIER) {
            if(!pointerTypeQualifiers.empty()) {
                parserState.error(parserState.previous(), "invalid type specifier");
            }
            else {
                typeSpecifiers.insert(parserState.previous().lexeme);
            }
        }
    } while(parserState.match({Token::Type::TYPE_QUALIFIER, Token::Type::TYPE_SPECIFIER, Token::Type::STAR}));

    // If no type specifiers are found
    if(typeSpecifiers.empty()) {
        parserState.error(parserState.peek(), "missing type specifier");
        throw ParseError();
    }
    // Lookup numeric type
    Type::ResolvedType type = getNumericType(typeSpecifiers, parserState.getContext());

    // If there are any type qualifiers, add const
    // **THINK** this relies of const being only qualifier
    if(!typeQualifiers.empty()) {
        type = type.addConst();
    }
    
    // Loop through levels of pointer indirection
    // **THINK** this relies of const being only qualifier
    for(const auto &p : pointerTypeQualifiers) {
        type = type.createPointer(!p.empty());
    }
    return type;
}

Expression::ExpressionPtr parsePrimary(ParserState &parserState)
{
    // primary-expression ::=
    //      identifier
    //      constant
    //      "(" expression ")"
    if (parserState.match({Token::Type::BOOLEAN, Token::Type::STRING,
                           Token::Type::DOUBLE_NUMBER, Token::Type::FLOAT_NUMBER, 
                           Token::Type::SCALAR_NUMBER, Token::Type::INT32_NUMBER, 
                           Token::Type::UINT32_NUMBER})) {
        return std::make_unique<Expression::Literal>(parserState.previous());
    }
    else if(parserState.match(Token::Type::IDENTIFIER)) {
        return std::make_unique<Expression::Identifier>(parserState.previous());
    }
    else if(parserState.match(Token::Type::LEFT_PAREN)) {
        auto expression = parseExpression(parserState);

        parserState.consume(Token::Type::RIGHT_PAREN, "Expect ')' after expression");
        return std::make_unique<Expression::Grouping>(std::move(expression));
    }

    parserState.error("Expect expression");
    throw ParseError();
}

Expression::ExpressionPtr parsePostfix(ParserState &parserState)
{
    // postfix-expression ::=
    //      primary-expression
    //      postfix-expression "[" expression "]"
    //      postfix-expression "(" argument-expression-list? ")"
    //      postfix-expression "++"
    //      postfix-expression "--"

    // argument-expression-list ::=
    //      assignment-expression
    //      argument-expression-list "," assignment-expression

    auto expression = parsePrimary(parserState);

    while(true) {
        // If this is a function call
        if(parserState.match(Token::Type::LEFT_PAREN)) {
            // Build list of arguments
            Expression::ExpressionList arguments;
            if(!parserState.check(Token::Type::RIGHT_PAREN)) {
                do {
                    arguments.emplace_back(parseAssignment(parserState));
                } while(parserState.match(Token::Type::COMMA));
            }

            Token closingParen = parserState.consume(Token::Type::RIGHT_PAREN,
                                                     "Expect ')' after arguments.");

            expression = std::make_unique<Expression::Call>(std::move(expression),
                                                            closingParen,
                                                            std::move(arguments));
        }
        // Otherwise, if this is an array index
        if(parserState.match(Token::Type::LEFT_SQUARE_BRACKET)) {
            auto index = parseExpression(parserState);
            Token closingSquareBracket = parserState.consume(Token::Type::RIGHT_SQUARE_BRACKET,
                                                             "Expect ']' after index.");

            expression = std::make_unique<Expression::ArraySubscript>(std::move(expression),
                                                                      closingSquareBracket,
                                                                      std::move(index));
        }
        // Otherwise if this is an increment or decrement
        else if(parserState.match({Token::Type::PLUS_PLUS, Token::Type::MINUS_MINUS})) {
            Token op = parserState.previous();

            // If expression is a valid l-value, 
            if(expression->isLValue()) {
                expression = std::make_unique<Expression::PostfixIncDec>(std::move(expression), op);
            }
            else {
                parserState.error(op, "Expression is not assignable");
            }
        }
        else {
            break;
        }
    }

    return expression;
}


Expression::ExpressionPtr parseUnary(ParserState &parserState)
{
    // unary-expression ::=
    //      postfix-expression
    //      "++" unary-expression
    //      "--" unary-expression
    //      "*" cast-expression
    //      "+" cast-expression
    //      "-" cast-expression
    //      "~" cast-expression
    //      "!" cast-expression
    //      "sizeof" unary-expression       **TODO** 
    //      "sizeof" "(" type-name ")"      **TODO** 
    if(parserState.match({Token::Type::STAR, Token::Type::PLUS, Token::Type::MINUS,
                          Token::Type::TILDA, Token::Type::NOT}))
    {
        Token op = parserState.previous();
        return std::make_unique<Expression::Unary>(op, parseCast(parserState));
    }
    else if(parserState.match({Token::Type::PLUS_PLUS, Token::Type::MINUS_MINUS})) {
        Token op = parserState.previous();
        auto expression = parseUnary(parserState);

        // If expression is a valid l-value, 
        if(expression->isLValue()) {
            return std::make_unique<Expression::PrefixIncDec>(std::move(expression), op);
        }
        else {
            parserState.error(op, "Expression is not assignable");
        }
    }

    return parsePostfix(parserState);
}

Expression::ExpressionPtr parseCast(ParserState &parserState)
{
    // cast-expression ::=
    //      unary-expression
    //      "(" type-name ")" cast-expression

    // If next token is a left parenthesis
    if(parserState.match(Token::Type::LEFT_PAREN)) {
        // If this is followed by some part of a type declarator
        if(parserState.match({Token::Type::TYPE_QUALIFIER, Token::Type::TYPE_SPECIFIER})) {
            // Parse declaration specifiers
            const auto type = parseDeclarationSpecifiers(parserState);

            const auto closingParen = parserState.consume(Token::Type::RIGHT_PAREN, "Expect ')' after cast type.");

            return std::make_unique<Expression::Cast>(type, parseCast(parserState), closingParen);
        }
        // Otherwise, rewind parser state so left parenthesis can be parsed again
        // **YUCK**
        else {
            parserState.rewind();
        }
    }

    return parseUnary(parserState);
}

Expression::ExpressionPtr parseMultiplicative(ParserState &parserState)
{
    // multiplicative-expression ::=
    //      cast-expression
    //      multiplicative-parseExpression "*" cast-parseExpression
    //      multiplicative-parseExpression "/" cast-parseExpression
    //      multiplicative-parseExpression "%" cast-parseExpression
    return parseBinary(parserState, parseCast, 
                       {Token::Type::STAR, Token::Type::SLASH, Token::Type::PERCENT});
}

Expression::ExpressionPtr parseAdditive(ParserState &parserState)
{
    // additive-expression ::=
    //      multiplicative-expression
    //      additive-parseExpression "+" multiplicative-parseExpression
    //      additive-parseExpression "-" multiplicative-parseExpression
    return parseBinary(parserState, parseMultiplicative, 
                       {Token::Type::MINUS, Token::Type::PLUS});
}

Expression::ExpressionPtr parseShift(ParserState &parserState)
{
    // shift-expression ::=
    //      additive-expression
    //      shift-parseExpression "<<" additive-parseExpression
    //      shift-parseExpression ">>" additive-parseExpression
    return parseBinary(parserState, parseAdditive, 
                       {Token::Type::SHIFT_LEFT, Token::Type::SHIFT_RIGHT});
}

Expression::ExpressionPtr parseRelational(ParserState &parserState)
{
    // relational-expression ::=
    //      shift-expression
    //      relational-parseExpression "<" shift-parseExpression
    //      relational-parseExpression ">" shift-parseExpression
    //      relational-parseExpression "<=" shift-parseExpression
    //      relational-parseExpression ">=" shift-parseExpression
    return parseBinary(parserState, parseShift, 
                       {Token::Type::GREATER, Token::Type::GREATER_EQUAL, 
                        Token::Type::LESS, Token::Type::LESS_EQUAL});
}

Expression::ExpressionPtr parseEquality(ParserState &parserState)
{
    // equality-expression ::=
    //      relational-expression
    //      equality-parseExpression "==" relational-parseExpression
    //      equality-parseExpression "!=" relational-parseExpression
    return parseBinary(parserState, parseRelational, 
                       {Token::Type::NOT_EQUAL, Token::Type::EQUAL_EQUAL});
}
Expression::ExpressionPtr parseAnd(ParserState &parserState)
{
    // AND-expression ::=
    //      equality-expression
    //      AND-expression "&" equality-expression
    return parseBinary(parserState, parseEquality, {Token::Type::AMPERSAND});
}

Expression::ExpressionPtr parseXor(ParserState &parserState)
{
    // exclusive-OR-expression ::=
    //      AND-expression
    //      exclusive-OR-expression "^" AND-expression
    return parseBinary(parserState, parseAnd, {Token::Type::CARET});
}

Expression::ExpressionPtr parseOr(ParserState &parserState)
{
    // inclusive-OR-expression ::=
    //      exclusive-OR-expression
    //      inclusive-OR-expression "|" exclusive-OR-expression
    return parseBinary(parserState, parseXor, {Token::Type::PIPE});
}

Expression::ExpressionPtr parseLogicalAnd(ParserState &parserState)
{
    // logical-AND-expression ::=
    //      inclusive-OR-expression
    //      logical-AND-expression "&&" inclusive-OR-expression
    // **THINK** parseLogicalAnd here (obviously) stack-overflows - why is this the grammar?
    auto expression = parseOr(parserState);

    while(parserState.match(Token::Type::AMPERSAND_AMPERSAND)) {
        Token op = parserState.previous();
        auto right = parseOr(parserState);
        expression = std::make_unique<Expression::Logical>(std::move(expression), op, std::move(right));
    }
    return expression;
}

Expression::ExpressionPtr parseLogicalOr(ParserState &parserState)
{
    // logical-OR-expression ::=
    //      logical-AND-expression
    //      logical-OR-expression "||" logical-AND-expression
    // **THINK** parseLogicalOr here (obviously) stack-overflows - why is this the grammar?
    auto expression = parseLogicalAnd(parserState);

    while(parserState.match(Token::Type::PIPE_PIPE)) {
        Token op = parserState.previous();
        auto right = parseLogicalAnd(parserState);
        expression = std::make_unique<Expression::Logical>(std::move(expression), op, std::move(right));
    }
    return expression;
}

Expression::ExpressionPtr parseConditional(ParserState &parserState)
{
    // conditional-expression ::=
    //      logical-OR-expression
    //      logical-OR-expression "?" expression ":" conditional-expression
    auto cond = parseLogicalOr(parserState);
    if(parserState.match(Token::Type::QUESTION)) {
        Token question = parserState.previous();
        auto trueExpression = parseExpression(parserState);
        parserState.consume(Token::Type::COLON, "Expect ':' in conditional expression.");
        auto falseExpression = parseConditional(parserState);
        return std::make_unique<Expression::Conditional>(std::move(cond), question, std::move(trueExpression), 
                                                         std::move(falseExpression));
    }

    return cond;
}

Expression::ExpressionPtr parseAssignment(ParserState &parserState)
{
    // assignment-expression ::=
    //      conditional-expression
    //      unary-expression assignment-operator assignment-expression
    auto expression = parseConditional(parserState);
    if(parserState.match({Token::Type::EQUAL, Token::Type::STAR_EQUAL, Token::Type::SLASH_EQUAL, 
                          Token::Type::PERCENT_EQUAL, Token::Type::PLUS_EQUAL, Token::Type::MINUS_EQUAL, 
                          Token::Type::AMPERSAND_EQUAL, Token::Type::CARET_EQUAL, Token::Type::PIPE_EQUAL,
                          Token::Type::SHIFT_LEFT_EQUAL, Token::Type::SHIFT_RIGHT_EQUAL})) 
    {
        Token op = parserState.previous();
        auto value = parseAssignment(parserState);

        // If expression is a valid l-value, 
        if(expression->isLValue()) {
            return std::make_unique<Expression::Assignment>(std::move(expression), op, std::move(value));
        }
        else {
            parserState.error(op, "Expression is not assignable");
        }
    }

    return expression;
}

Expression::ExpressionPtr parseExpression(ParserState &parserState)
{
    // expression ::=
    //      assignment-expression
    //      expression "," assignment-expression
    return parseBinary(parserState, parseAssignment, 
                       {Token::Type::COMMA});
}

Statement::StatementPtr parseLabelledStatement(ParserState &parserState)
{
    // labeled-statement ::=
    //      "case" constant-expression ":" statement
    //      "default" ":" statement
    const auto keyword = parserState.previous();

    Expression::ExpressionPtr value;
    if(keyword.type == Token::Type::CASE) {
        value = parseConditional(parserState);
    }

    parserState.consume(Token::Type::COLON, "Expect ':' after labelled statement."); 
 
    return std::make_unique<Statement::Labelled>(keyword, std::move(value), 
                                                 parseStatement(parserState));
}

Statement::StatementPtr parseCompoundStatement(ParserState &parserState)
{
    // compound-statement ::=
    //      "{" block-item-list? "}"
    // block-item-list ::=
    //      block-item
    //      block-item-list block-item
    // block-item ::=
    //      declaration
    //      statement
    Statement::StatementList statements;
    while(!parserState.check(Token::Type::RIGHT_BRACE) && !parserState.isAtEnd()) {
        statements.emplace_back(parseBlockItem(parserState));
    }
    parserState.consume(Token::Type::RIGHT_BRACE, "Expect '}' after compound statement.");

    return std::make_unique<Statement::Compound>(std::move(statements));
}

Statement::StatementPtr parseExpressionStatement(ParserState &parserState)
{
    //  expression-statement ::=
    //      expression? ";"
    if(parserState.match(Token::Type::SEMICOLON)) {
        return std::make_unique<Statement::Expression>(nullptr);
    }
    else {
        auto expression = parseExpression(parserState);
    
        parserState.consume(Token::Type::SEMICOLON, "Expect ';' after expression");
        return std::make_unique<Statement::Expression>(std::move(expression));
    }
}

Statement::StatementPtr parseSelectionStatement(ParserState &parserState)
{
    // selection-statement ::=
    //      "if" "(" expression ")" statement
    //      "if" "(" expression ")" statement "else" statement
    //      "switch" "(" expression ")" compound-statement
    const auto keyword = parserState.previous();
    parserState.consume(Token::Type::LEFT_PAREN, "Expect '(' after '" + keyword.lexeme + "'");
    auto condition = parseExpression(parserState);
    parserState.consume(Token::Type::RIGHT_PAREN, "Expect ')' after '" + keyword.lexeme + "'");

    // If this is an if statement
    if(keyword.type == Token::Type::IF) {
        auto thenBranch = parseStatement(parserState);
        Statement::StatementPtr elseBranch;
        if(parserState.match(Token::Type::ELSE)) {
            elseBranch = parseStatement(parserState);
        }

        return std::make_unique<Statement::If>(keyword, std::move(condition),
                                               std::move(thenBranch),
                                               std::move(elseBranch));
    }
    // Otherwise (switch statement)
    else {
        // **NOTE** this is a slight simplification of the C standard where any type of statement can be used as the body of the switch
        parserState.consume(Token::Type::LEFT_BRACE, "Expect '{' after switch statement.");
        auto body = parseCompoundStatement(parserState);
        return std::make_unique<Statement::Switch>(keyword, std::move(condition), std::move(body));
    }
}

Statement::StatementPtr parseIterationStatement(ParserState &parserState)
{
    // iteration-statement ::=
    //      "while" "(" expression ")" statement
    //      "do" statement "while" "(" expression ")" ";"
    //      "for" statement
    //      "for" "(" expression? ";" expression? ";" expression? ")" statement
    //      "for" "(" declaration expression? ";" expression? ")" statement
    //      "for_each_synapse" statement

    // If this is a while statement
    if(parserState.previous().type == Token::Type::WHILE) {
        const auto whileToken = parserState.previous();
        parserState.consume(Token::Type::LEFT_PAREN, "Expect '(' after 'while'");
        auto condition = parseExpression(parserState);
        parserState.consume(Token::Type::RIGHT_PAREN, "Expect ')' after 'while'");
        auto body = parseStatement(parserState);

        return std::make_unique<Statement::While>(whileToken, std::move(condition), 
                                                  std::move(body));
    }
    // Otherwise, if this is a do statement 
    else if(parserState.previous().type == Token::Type::DO) {
        auto body = parseStatement(parserState);
        parserState.consume(Token::Type::WHILE, "Expected 'while' after 'do' statement body");
        const auto whileToken = parserState.previous();
        parserState.consume(Token::Type::LEFT_PAREN, "Expect '(' after 'while'");
        auto condition = parseExpression(parserState);
        parserState.consume(Token::Type::RIGHT_PAREN, "Expect ')' after 'while'");
        parserState.consume(Token::Type::SEMICOLON, "Expect ';' after while");
        return std::make_unique<Statement::Do>(whileToken, std::move(condition), 
                                               std::move(body));
    }
    // Otherwise, if this is a for_each_synapse statement
    else if(parserState.previous().type == Token::Type::FOR_EACH_SYNAPSE) {
        const auto forEachSynapse = parserState.previous();
        auto body = parseStatement(parserState);
        return std::make_unique<Statement::ForEachSynapse>(forEachSynapse,
                                                           std::move(body));
    }
    // Otherwise, it's a for statement
    else {
        const auto forToken = parserState.previous();
        parserState.consume(Token::Type::LEFT_PAREN, "Expect '(' after 'for'");

        // If statement starts with a semicolon - no initialiser
        Statement::StatementPtr initialiser;
        if(parserState.match(Token::Type::SEMICOLON)) {
            initialiser = nullptr;
        }
        // Otherwise, if it starts with a declaration
        else if(parserState.match({Token::Type::TYPE_SPECIFIER, Token::Type::TYPE_QUALIFIER})) {
            initialiser = parseDeclaration(parserState);
        }
        // Otherwise, must be expression (statement consumes semicolon)
        else {
            initialiser = parseExpressionStatement(parserState);
        }

        // Parse condition
        Expression::ExpressionPtr condition = nullptr;
        if(!parserState.check(Token::Type::SEMICOLON)) {
            condition = parseExpression(parserState);
        }
        parserState.consume(Token::Type::SEMICOLON, "Expect ';' after loop condition");

        // Parse increment
        Expression::ExpressionPtr increment = nullptr;
        if(!parserState.check(Token::Type::RIGHT_PAREN)) {
            increment = parseExpression(parserState);
        }
        parserState.consume(Token::Type::RIGHT_PAREN, "Expect ')' after for clauses");

        auto body = parseStatement(parserState);

        // Return for statement
        // **NOTE** we could "de-sugar" into a while statement but this makes pretty-printing easier
        return std::make_unique<Statement::For>(forToken, std::move(initialiser), 
                                                std::move(condition),
                                                std::move(increment),
                                                std::move(body));
    }
}

Statement::StatementPtr parseJumpStatement(ParserState &parserState)
{
    // jump-statement ::=
    //      "continue" ";"
    //      "break" ";"
    //      "return" expression? ";"    // **TODO**
    const Token token = parserState.previous();
    if(token.type == Token::Type::CONTINUE) {
        parserState.consume(Token::Type::SEMICOLON, "Expect ';' after continue");
        return std::make_unique<Statement::Continue>(token);
    }
    else if(token.type == Token::Type::BREAK) {
        parserState.consume(Token::Type::SEMICOLON, "Expect ';' after break");
        return std::make_unique<Statement::Break>(token);
    }
    // Otherwise (return statement)
    else {
        assert(false);
        return nullptr;
    }
}

Statement::StatementPtr parseStatement(ParserState &parserState)
{
    // statement ::=
    //      labeled-statement
    //      compound-statement
    //      expression-statement
    //      selection-statement     
    //      iteration-statement
    //      jump-statement
    if(parserState.match({Token::Type::CASE, Token::Type::DEFAULT})) {
        return parseLabelledStatement(parserState);
    }
    else if(parserState.match(Token::Type::LEFT_BRACE)) {
        return parseCompoundStatement(parserState);
    }
    else if(parserState.match({Token::Type::IF, Token::Type::SWITCH})) {
        return parseSelectionStatement(parserState);
    }
    else if(parserState.match({Token::Type::FOR, Token::Type::FOR_EACH_SYNAPSE, 
                              Token::Type::WHILE, Token::Type::DO})) {
        return parseIterationStatement(parserState);
    }
    else if(parserState.match({Token::Type::CONTINUE, Token::Type::BREAK})) {
        return parseJumpStatement(parserState);
    }
    else {
        return parseExpressionStatement(parserState);
    }
}

Statement::StatementPtr parseDeclaration(ParserState &parserState)
{
    // declaration ::=
    //      declaration-specifiers init-declarator-list? ";"

    // declaration-specifiers ::=
    //      declaration-specifiers?
    //      type-specifier declaration-specifiers?
    //      type-qualifier declaration-specifiers?

    // type-specifier ::=
    //      "char"
    //      "short"
    //      "int"
    //      "long"
    //      "float"
    //      "double"
    //      "scalar"
    //      "signed"
    //      "unsigned"
    //      "bool"
    //      typedef-name    // **TODO** not sure how to address ambiguity with subsequent identifier

    // type-qualifier ::=
    //      "const"

    // Parse declaration specifiers
    const auto type = parseDeclarationSpecifiers(parserState);

    // Read init declarator list
    std::vector<std::tuple<Token, Expression::ExpressionPtr>> initDeclaratorList;
    do {
        // init-declarator-list ::=
        //      init-declarator
        //      init-declarator-list "," init-declarator

        // init-declarator ::=
        //      declarator
        //      declarator "=" assignment-expression

        // declarator ::=
        //      identifier
        Token identifier = parserState.consume(Token::Type::IDENTIFIER, "Expect variable name");
        Expression::ExpressionPtr initialiser;
        if(parserState.match(Token::Type::EQUAL)) {
            initialiser = parseAssignment(parserState);
        }
        initDeclaratorList.emplace_back(identifier, std::move(initialiser));
    } while(!parserState.isAtEnd() && parserState.match(Token::Type::COMMA));

    parserState.consume(Token::Type::SEMICOLON, "Expect ';' after variable declaration");
    return std::make_unique<Statement::VarDeclaration>(type, std::move(initDeclaratorList));
}

std::unique_ptr<const Statement::Base> parseBlockItem(ParserState &parserState)
{
    // block-item ::=
    //      declaration
    //      statement
    try {
        if(parserState.match({Token::Type::TYPE_SPECIFIER, Token::Type::TYPE_QUALIFIER})) {
            return parseDeclaration(parserState);
        }
        else {
            return parseStatement(parserState);
        }
    }
    catch(ParseError &) {
        synchronise(parserState);
        return nullptr;
    }
}
}   // Anonymous namespace


//---------------------------------------------------------------------------
// GeNN::Transpiler::Parser
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Parser
{
Expression::ExpressionPtr parseExpression(const std::vector<Token> &tokens, const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    // Parse expression
    ParserState parserState(tokens, context, errorHandler);

    try {
        auto expression = parseExpression(parserState);
    
        // If there are more tokens, raise error
        if(!parserState.isAtEnd()) {
            parserState.error(parserState.peek(), "Unexpected token after expression");
        }
    
        return expression;
    }
    catch(ParseError &) {
        return nullptr;
    }
}
//---------------------------------------------------------------------------
Statement::StatementList parseBlockItemList(const std::vector<Token> &tokens, const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    ParserState parserState(tokens, context, errorHandler);
    std::vector<std::unique_ptr<const Statement::Base>> statements;

    while(!parserState.isAtEnd()) {
        statements.emplace_back(parseBlockItem(parserState));
    }
    return statements;
}
//---------------------------------------------------------------------------
const GeNN::Type::ResolvedType parseNumericType(const std::vector<Token> &tokens, const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    // Parse type specifiers
    ParserState parserState(tokens, context, errorHandler);
    std::multiset<std::string> typeSpecifiers;
    while(parserState.match(Token::Type::TYPE_SPECIFIER)) {
        typeSpecifiers.insert(parserState.previous().lexeme);
    };

    // If there are more tokens, raise error
    if(!parserState.isAtEnd()) {
        parserState.error(parserState.peek(), "Unexpected token after type");
    }
    
    // Return numeric type
    return getNumericType(typeSpecifiers, context);
}
}
