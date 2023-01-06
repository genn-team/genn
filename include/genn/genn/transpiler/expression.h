#pragma once

// Standard C++ includes
#include <memory>
#include <vector>

// Transpiler includes
#include "transpiler/token.h"

// Forward declarations
namespace GeNN::Transpiler::Expression 
{
class Visitor;
}
namespace GeNN::Type
{
class Base;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Base
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Expression
{
class Base
{
public:
    virtual void accept(Visitor &visitor) const = 0;
};

typedef std::unique_ptr<Base const> ExpressionPtr;
typedef std::vector<ExpressionPtr> ExpressionList;

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::ArraySubscript
//---------------------------------------------------------------------------
class ArraySubscript : public Base
{
public:
    ArraySubscript(Token pointerName, ExpressionPtr index)
    :  m_PointerName(pointerName), m_Index(std::move(index))
    {}

    virtual void accept(Visitor &visitor) const final;

    const Token &getPointerName() const { return m_PointerName; }
    const ExpressionPtr &getIndex() const { return m_Index; }

private:
    const Token m_PointerName;
    const ExpressionPtr m_Index;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Assignment
//---------------------------------------------------------------------------
class Assignment : public Base
{
public:
    Assignment(Token varName, Token op, ExpressionPtr value)
    :  m_VarName(varName), m_Operator(op), m_Value(std::move(value))
    {}

    virtual void accept(Visitor &visitor) const final;

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
class Binary : public Base
{
public:
    Binary(ExpressionPtr left, Token op, ExpressionPtr right)
    :  m_Left(std::move(left)), m_Operator(op), m_Right(std::move(right))
    {}

    virtual void accept(Visitor &visitor) const final;

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
class Call : public Base
{
public:
    Call(ExpressionPtr callee, Token closingParen, ExpressionList arguments)
    :  m_Callee(std::move(callee)), m_ClosingParen(closingParen), m_Arguments(std::move(arguments))
    {}

    virtual void accept(Visitor &visitor) const final;

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
class Cast : public Base
{
public:
    Cast(const Type::Base *type, bool isConst, ExpressionPtr expression)
    :  m_Type(type), m_Const(isConst), m_Expression(std::move(expression))
    {}

    virtual void accept(Visitor &visitor) const final;

    const Base *getExpression() const { return m_Expression.get(); }
    
    const Type::Base *getType() const { return m_Type; }
    bool isConst() const { return m_Const; }

private:
    const Type::Base *m_Type;
    bool m_Const;
    const ExpressionPtr m_Expression;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Conditional
//---------------------------------------------------------------------------
class Conditional : public Base
{
public:
    Conditional(ExpressionPtr condition, Token question, ExpressionPtr trueExpression, ExpressionPtr falseExpression)
    :  m_Condition(std::move(condition)), m_Question(question), m_True(std::move(trueExpression)), m_False(std::move(falseExpression))
    {}

    virtual void accept(Visitor &visitor) const final;

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
class Grouping : public Base
{
public:
    Grouping(ExpressionPtr expression)
    :  m_Expression(std::move(expression))
    {}

    virtual void accept(Visitor &visitor) const final;

    const Base *getExpression() const { return m_Expression.get(); }

private:
    const ExpressionPtr m_Expression;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Literal
//---------------------------------------------------------------------------
class Literal : public Base
{
public:
    Literal(Token::LiteralValue value)
    :  m_Value(value)
    {}

    virtual void accept(Visitor &visitor) const final;

    Token::LiteralValue getValue() const { return m_Value; }

private:
    const Token::LiteralValue m_Value;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Logical
//---------------------------------------------------------------------------
class Logical : public Base
{
public:
    Logical(ExpressionPtr left, Token op, ExpressionPtr right)
    :  m_Left(std::move(left)), m_Operator(op), m_Right(std::move(right))
    {}

    virtual void accept(Visitor &visitor) const final;

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
class PostfixIncDec : public Base
{
public:
    PostfixIncDec(Token varName, Token op)
    :  m_VarName(varName), m_Operator(op)
    {}

    virtual void accept(Visitor &visitor) const final;

    const Token &getVarName() const { return m_VarName; }
    const Token &getOperator() const { return m_Operator; }

private:
    const Token m_VarName;
    const Token m_Operator;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::PrefixIncDec
//---------------------------------------------------------------------------
class PrefixIncDec : public Base
{
public:
    PrefixIncDec(Token varName, Token op)
    :  m_VarName(varName), m_Operator(op)
    {}

    virtual void accept(Visitor &visitor) const final;

    const Token &getVarName() const { return m_VarName; }
    const Token &getOperator() const { return m_Operator; }

private:
    const Token m_VarName;
    const Token m_Operator;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Variable
//---------------------------------------------------------------------------
class Variable : public Base
{
public:
    Variable(Token name)
    :  m_Name(name)
    {}

    virtual void accept(Visitor &visitor) const final;

    const Token &getName() const { return m_Name; }

private:
    const Token m_Name;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Unary
//---------------------------------------------------------------------------
class Unary : public Base
{
public:
    Unary(Token op, ExpressionPtr right)
    :  m_Operator(op), m_Right(std::move(right))
    {}

    virtual void accept(Visitor &visitor) const final;

    const Token &getOperator() const { return m_Operator; }
    const Base *getRight() const { return m_Right.get(); }

private:
    const Token m_Operator;
    const ExpressionPtr m_Right;
};


//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Visitor
//---------------------------------------------------------------------------
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
}   // namespace GeNN::Transpiler::Expression