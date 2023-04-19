#pragma once

// Standard C++ includes
#include <memory>
#include <vector>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/token.h"

// Forward declarations
namespace GeNN::Transpiler::Expression 
{
class ArraySubscript;
class Assignment;
class Binary;
class Call;
class Cast;
class Conditional;
class Grouping;
class Literal;
class Logical;
class PostfixIncDec;
class PrefixIncDec;
class Variable;
class Unary;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Visitor
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Expression
{
class Visitor
{
public:
    virtual void visit(const ArraySubscript &arraySubscript) = 0;
    virtual void visit(const Assignment &assignement) = 0;
    virtual void visit(const Binary &binary) = 0;
    virtual void visit(const Call &call) = 0;
    virtual void visit(const Cast &cast) = 0;
    virtual void visit(const Conditional &conditional) = 0;
    virtual void visit(const Grouping &grouping) = 0;
    virtual void visit(const Literal &literal) = 0;
    virtual void visit(const Logical &logical) = 0;
    virtual void visit(const PostfixIncDec &postfixIncDec) = 0;
    virtual void visit(const PrefixIncDec &postfixIncDec) = 0;
    virtual void visit(const Variable &variable) = 0;
    virtual void visit(const Unary &unary) = 0;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Base
//---------------------------------------------------------------------------
class Base
{
public:
    virtual void accept(Visitor &visitor) const = 0;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Acceptable
//---------------------------------------------------------------------------
template<typename T>
class Acceptable : public Base
{
public:
    virtual void accept(Visitor &visitor) const final
    {
        visitor.visit(static_cast<const T&>(*this));
    }
};

typedef std::unique_ptr<Base const> ExpressionPtr;
typedef std::vector<ExpressionPtr> ExpressionList;

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::ArraySubscript
//---------------------------------------------------------------------------
class ArraySubscript : public Acceptable<ArraySubscript>
{
public:
    ArraySubscript(ExpressionPtr array, Token closingSquareBracket, ExpressionPtr index)
    :  m_Array(std::move(array)), m_ClosingSquareBracket(closingSquareBracket), m_Index(std::move(index))
    {}

    const Base *getArray() const { return m_Array.get(); }
    const Token &getClosingSquareBracket() const { return m_ClosingSquareBracket; }
    const Base *getIndex() const { return m_Index.get(); }

private:
    const ExpressionPtr m_Array;
    const Token m_ClosingSquareBracket;
    const ExpressionPtr m_Index;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Assignment
//---------------------------------------------------------------------------
class Assignment : public Acceptable<Assignment>
{
public:
    Assignment(Token varName, Token op, ExpressionPtr value)
    :  m_VarName(varName), m_Operator(op), m_Value(std::move(value))
    {}

    const Token &getVarName() const { return m_VarName; }
    const Token &getOperator() const { return m_Operator; }
    const Base *getValue() const { return m_Value.get(); }

private:
    const Token m_VarName;
    const Token m_Operator;
    const ExpressionPtr m_Value;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Binary
//---------------------------------------------------------------------------
class Binary : public Acceptable<Binary>
{
public:
    Binary(ExpressionPtr left, Token op, ExpressionPtr right)
    :  m_Left(std::move(left)), m_Operator(op), m_Right(std::move(right))
    {}

    const Base *getLeft() const { return m_Left.get(); }
    const Token &getOperator() const { return m_Operator; }
    const Base *getRight() const { return m_Right.get(); }

private:
    const ExpressionPtr m_Left;
    const Token m_Operator;
    const ExpressionPtr m_Right;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Call
//---------------------------------------------------------------------------
class Call : public Acceptable<Call>
{
public:
    Call(ExpressionPtr callee, Token closingParen, ExpressionList arguments)
    :  m_Callee(std::move(callee)), m_ClosingParen(closingParen), m_Arguments(std::move(arguments))
    {}

    const Base *getCallee() const { return m_Callee.get(); }
    const Token &getClosingParen() const { return m_ClosingParen; }
    const ExpressionList &getArguments() const { return m_Arguments; }

private:
    const ExpressionPtr m_Callee;
    const Token m_ClosingParen;
    const ExpressionList m_Arguments;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Cast
//---------------------------------------------------------------------------
class Cast : public Acceptable<Cast>
{
public:
    Cast(const Type::Base *type, ExpressionPtr expression, Token closingParen)
    :  m_Type(type), m_Expression(std::move(expression)), m_ClosingParen(closingParen)
    {}

    const Type::Base *getType() const{ return m_Type; }
    const Base *getExpression() const { return m_Expression.get(); }
    const Token &getClosingParen() const { return m_ClosingParen; }
    
private:
    const Type::Base *m_Type;
    const ExpressionPtr m_Expression;
    const Token m_ClosingParen;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Conditional
//---------------------------------------------------------------------------
class Conditional : public Acceptable<Conditional>
{
public:
    Conditional(ExpressionPtr condition, Token question, ExpressionPtr trueExpression, ExpressionPtr falseExpression)
    :  m_Condition(std::move(condition)), m_Question(question), m_True(std::move(trueExpression)), m_False(std::move(falseExpression))
    {}

    const Base *getCondition() const { return m_Condition.get(); }
    const Token &getQuestion() const { return m_Question; }
    const Base *getTrue() const { return m_True.get(); }
    const Base *getFalse() const { return m_False.get(); }

private:
    const ExpressionPtr m_Condition;
    const Token m_Question;
    const ExpressionPtr m_True;
    const ExpressionPtr m_False;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Grouping
//---------------------------------------------------------------------------
class Grouping : public Acceptable<Grouping>
{
public:
    Grouping(ExpressionPtr expression)
    :  m_Expression(std::move(expression))
    {}

    const Base *getExpression() const { return m_Expression.get(); }

private:
    const ExpressionPtr m_Expression;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Literal
//---------------------------------------------------------------------------
class Literal : public Acceptable<Literal>
{
public:
    Literal(Token value)
    :  m_Value(value)
    {}

    Token getValue() const { return m_Value; }

private:
    const Token m_Value;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Logical
//---------------------------------------------------------------------------
class Logical : public Acceptable<Logical>
{
public:
    Logical(ExpressionPtr left, Token op, ExpressionPtr right)
    :  m_Left(std::move(left)), m_Operator(op), m_Right(std::move(right))
    {}

    const Base *getLeft() const { return m_Left.get(); }
    const Token &getOperator() const { return m_Operator; }
    const Base *getRight() const { return m_Right.get(); }

private:
    const ExpressionPtr m_Left;
    const Token m_Operator;
    const ExpressionPtr m_Right;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::PostfixIncDec
//---------------------------------------------------------------------------
class PostfixIncDec : public Acceptable<PostfixIncDec>
{
public:
    PostfixIncDec(Token varName, Token op)
    :  m_VarName(varName), m_Operator(op)
    {}

    const Token &getVarName() const { return m_VarName; }
    const Token &getOperator() const { return m_Operator; }

private:
    const Token m_VarName;
    const Token m_Operator;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::PrefixIncDec
//---------------------------------------------------------------------------
class PrefixIncDec : public Acceptable<PrefixIncDec>
{
public:
    PrefixIncDec(Token varName, Token op)
    :  m_VarName(varName), m_Operator(op)
    {}

    const Token &getVarName() const { return m_VarName; }
    const Token &getOperator() const { return m_Operator; }

private:
    const Token m_VarName;
    const Token m_Operator;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Variable
//---------------------------------------------------------------------------
class Variable : public Acceptable<Variable>
{
public:
    Variable(Token name)
    :  m_Name(name)
    {}

    const Token &getName() const { return m_Name; }

private:
    const Token m_Name;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Unary
//---------------------------------------------------------------------------
class Unary : public Acceptable<Unary>
{
public:
    Unary(Token op, ExpressionPtr right)
    :  m_Operator(op), m_Right(std::move(right))
    {}

    const Token &getOperator() const { return m_Operator; }
    const Base *getRight() const { return m_Right.get(); }

private:
    const Token m_Operator;
    const ExpressionPtr m_Right;
};
}   // namespace GeNN::Transpiler::Expression