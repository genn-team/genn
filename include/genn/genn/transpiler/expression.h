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
template<typename>
class ArraySubscript;

template<typename>
class Assignment;

template<typename>
class Binary;

template<typename>
class Call;

template<typename>
class Cast;

template<typename>
class Conditional;

template<typename>
class Grouping;

template<typename>
class Literal;

template<typename>
class Logical;

template<typename>
class PostfixIncDec;

template<typename>
class PrefixIncDec;

template<typename>
class Identifier;

template<typename>
class Unary;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::NoAnnotation
//---------------------------------------------------------------------------
//! Default AST annotation type - no annotation!
namespace GeNN::Transpiler::Expression
{
class NoAnnotation
{
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Visitor
//---------------------------------------------------------------------------
template<typename A = NoAnnotation>
class Visitor
{
public:
    virtual void visit(const ArraySubscript<A> &arraySubscript) = 0;
    virtual void visit(const Assignment<A> &assignement) = 0;
    virtual void visit(const Binary<A> &binary) = 0;
    virtual void visit(const Call<A> &call) = 0;
    virtual void visit(const Cast<A> &cast) = 0;
    virtual void visit(const Conditional<A> &conditional) = 0;
    virtual void visit(const Grouping<A> &grouping) = 0;
    virtual void visit(const Literal<A> &literal) = 0;
    virtual void visit(const Logical<A> &logical) = 0;
    virtual void visit(const PostfixIncDec<A> &postfixIncDec) = 0;
    virtual void visit(const PrefixIncDec<A> &postfixIncDec) = 0;
    virtual void visit(const Identifier<A> &variable) = 0;
    virtual void visit(const Unary<A> &unary) = 0;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Base
//---------------------------------------------------------------------------
template<typename A = NoAnnotation>
class Base : public A
{
public:
    using A::A;
    virtual ~Base(){}

    virtual void accept(Visitor<A> &visitor) const = 0;
    virtual bool isLValue() const{ return false; }
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Acceptable
//---------------------------------------------------------------------------
template<typename T, typename A = NoAnnotation>
class Acceptable : public Base<A>
{
public:
    using Base<A>::Base;

    virtual void accept(Visitor<A> &visitor) const final
    {
        visitor.visit(static_cast<const T&>(*this));
    }
};

template<typename A = NoAnnotation>
using ExpressionPtr = std::unique_ptr<Base<A> const> ;

template<typename A = NoAnnotation>
using ExpressionList = std::vector<ExpressionPtr<A>>;

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::ArraySubscript
//---------------------------------------------------------------------------
template<typename A>
class ArraySubscript : public Acceptable<ArraySubscript<A>, A>
{
public:
    template<typename... AnnotationArgs>
    ArraySubscript(ExpressionPtr<A> array, Token closingSquareBracket,
                   ExpressionPtr<A> index, AnnotationArgs&&... annotationArgs)
    :   Acceptable<ArraySubscript<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...), 
        m_Array(std::move(array)), m_ClosingSquareBracket(closingSquareBracket), 
        m_Index(std::move(index))
    {}

    //------------------------------------------------------------------------
    // Expression::Base virtuals
    //------------------------------------------------------------------------
    virtual bool isLValue() const{ return true; }

    const auto *getArray() const { return m_Array.get(); }
    const auto &getClosingSquareBracket() const { return m_ClosingSquareBracket; }
    const auto *getIndex() const { return m_Index.get(); }

private:
    ExpressionPtr<A> m_Array;
    Token m_ClosingSquareBracket;
    ExpressionPtr<A> m_Index;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Assignment
//---------------------------------------------------------------------------
template<typename A>
class Assignment : public Acceptable<Assignment<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Assignment(ExpressionPtr<A> assignee, Token op, ExpressionPtr<A> value, 
               AnnotationArgs&&... annotationArgs)
    :  Acceptable<Assignment<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Assignee(std::move(assignee)), m_Operator(op), m_Value(std::move(value))
    {}

    const auto *getAssignee() const { return m_Assignee.get(); }
    const auto &getOperator() const { return m_Operator; }
    const auto *getValue() const { return m_Value.get(); }

private:
    ExpressionPtr<A> m_Assignee;
    Token m_Operator;
    ExpressionPtr<A> m_Value;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Binary
//---------------------------------------------------------------------------
template<typename A>
class Binary : public Acceptable<Binary<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Binary(ExpressionPtr<A> left, Token op, ExpressionPtr<A> right,
           AnnotationArgs&&... annotationArgs)
    :  Acceptable<Binary<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Left(std::move(left)), m_Operator(op), m_Right(std::move(right))
    {}

    const auto *getLeft() const { return m_Left.get(); }
    const auto &getOperator() const { return m_Operator; }
    const auto *getRight() const { return m_Right.get(); }

private:
    ExpressionPtr<A> m_Left;
    Token m_Operator;
    ExpressionPtr<A> m_Right;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Call
//---------------------------------------------------------------------------
template<typename A>
class Call : public Acceptable<Call<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Call(ExpressionPtr<A> callee, Token closingParen, ExpressionList<A> arguments,
         AnnotationArgs&&... annotationArgs)
    :   Acceptable<Call<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Callee(std::move(callee)), m_ClosingParen(closingParen), m_Arguments(std::move(arguments))
    {}

    const auto *getCallee() const { return m_Callee.get(); }
    const auto &getClosingParen() const { return m_ClosingParen; }
    const auto &getArguments() const { return m_Arguments; }

private:
    ExpressionPtr<A> m_Callee;
    Token m_ClosingParen;
    ExpressionList<A> m_Arguments;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Cast
//---------------------------------------------------------------------------
template<typename A>
class Cast : public Acceptable<Cast<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Cast(const Type::ResolvedType &type, ExpressionPtr<A> expression,
         Token closingParen, AnnotationArgs&&... annotationArgs)
    :   Acceptable<Cast<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Type(type), m_Expression(std::move(expression)), m_ClosingParen(closingParen)
    {}

    const auto &getType() const{ return m_Type; }
    const auto *getExpression() const { return m_Expression.get(); }
    const auto &getClosingParen() const { return m_ClosingParen; }
    
private:
    Type::ResolvedType m_Type;
    ExpressionPtr<A> m_Expression;
    Token m_ClosingParen;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Conditional
//---------------------------------------------------------------------------
template<typename A>
class Conditional : public Acceptable<Conditional<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Conditional(ExpressionPtr<A> condition, Token question, ExpressionPtr<A> trueExpression, 
                ExpressionPtr<A> falseExpression, AnnotationArgs&&... annotationArgs)
    :   Acceptable<Conditional<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Condition(std::move(condition)), m_Question(question), m_True(std::move(trueExpression)), m_False(std::move(falseExpression))
    {}

    const auto *getCondition() const { return m_Condition.get(); }
    const auto &getQuestion() const { return m_Question; }
    const auto *getTrue() const { return m_True.get(); }
    const auto *getFalse() const { return m_False.get(); }

private:
    ExpressionPtr<A> m_Condition;
    Token m_Question;
    ExpressionPtr<A> m_True;
    ExpressionPtr<A> m_False;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Grouping
//---------------------------------------------------------------------------
template<typename A>
class Grouping : public Acceptable<Grouping<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Grouping(ExpressionPtr<A> expression, AnnotationArgs&&... annotationArgs)
    :   Acceptable<Grouping<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Expression(std::move(expression))
    {}

    //------------------------------------------------------------------------
    // Expression::Base virtuals
    //------------------------------------------------------------------------
    virtual bool isLValue() const{ return m_Expression->isLValue(); }

    const auto *getExpression() const { return m_Expression.get(); }

private:
    ExpressionPtr<A> m_Expression;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Literal
//---------------------------------------------------------------------------
template<typename A>
class Literal : public Acceptable<Literal<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Literal(Token value, AnnotationArgs&&... annotationArgs)
    :   Acceptable<Literal<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Value(value)
    {}

    //------------------------------------------------------------------------
    // Expression::Base virtuals
    //------------------------------------------------------------------------
    virtual bool isLValue() const{ return (m_Value.type == Token::Type::STRING); }

    const auto &getValue() const { return m_Value; }

private:
    Token m_Value;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Logical
//---------------------------------------------------------------------------
template<typename A>
class Logical : public Acceptable<Logical<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Logical(ExpressionPtr<A> left, Token op, ExpressionPtr<A> right,
            AnnotationArgs&&... annotationArgs)
    :   Acceptable<Logical<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Left(std::move(left)), m_Operator(op), m_Right(std::move(right))
    {}

    const auto *getLeft() const { return m_Left.get(); }
    const auto &getOperator() const { return m_Operator; }
    const auto *getRight() const { return m_Right.get(); }

private:
    ExpressionPtr<A> m_Left;
    Token m_Operator;
    ExpressionPtr<A> m_Right;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::PostfixIncDec
//---------------------------------------------------------------------------
template<typename A>
class PostfixIncDec : public Acceptable<PostfixIncDec<A>, A>
{
public:
    template<typename... AnnotationArgs>
    PostfixIncDec(ExpressionPtr<A> target, Token op, AnnotationArgs&&... annotationArgs)
    :   Acceptable<PostfixIncDec<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Target(std::move(target)), m_Operator(op)
    {}

    const auto *getTarget() const { return m_Target.get(); }
    const auto &getOperator() const { return m_Operator; }

private:
    ExpressionPtr<A> m_Target;
    Token m_Operator;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::PrefixIncDec
//---------------------------------------------------------------------------
template<typename A>
class PrefixIncDec : public Acceptable<PrefixIncDec<A>, A>
{
public:
    template<typename... AnnotationArgs>
    PrefixIncDec(ExpressionPtr<A> target, Token op, AnnotationArgs&&... annotationArgs)
    :   Acceptable<PrefixIncDec<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Target(std::move(target)), m_Operator(op)
    {}

    const auto *getTarget() const { return m_Target.get(); }
    const auto &getOperator() const { return m_Operator; }

private:
    ExpressionPtr<A> m_Target;
    Token m_Operator;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Identifier
//---------------------------------------------------------------------------
template<typename A>
class Identifier : public Acceptable<Identifier<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Identifier(Token name, AnnotationArgs&&... annotationArgs)
    :   Acceptable<Identifier<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Name(name)
    {}

    //------------------------------------------------------------------------
    // Expression::Base virtuals
    //------------------------------------------------------------------------
    virtual bool isLValue() const{ return true; }

    const auto &getName() const { return m_Name; }

private:
    Token m_Name;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Expression::Unary
//---------------------------------------------------------------------------
template<typename A>
class Unary : public Acceptable<Unary<A>, A>
{
public:
    template<typename... AnnotationArgs>
    Unary(Token op, ExpressionPtr<A> right, AnnotationArgs&&... annotationArgs)
    :   Acceptable<Unary<A>, A>(std::forward<AnnotationArgs>(annotationArgs)...),
        m_Operator(op), m_Right(std::move(right))
    {}

    //------------------------------------------------------------------------
    // Expression::Base virtuals
    //------------------------------------------------------------------------
    virtual bool isLValue() const{ return (m_Operator.type == Token::Type::STAR); }

    const auto &getOperator() const { return m_Operator; }
    const auto *getRight() const { return m_Right.get(); }

private:
    Token m_Operator;
    ExpressionPtr<A> m_Right;
};
}   // namespace GeNN::Transpiler::Expression
