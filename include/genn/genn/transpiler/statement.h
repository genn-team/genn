#pragma once

// Standard C++ includes
#include <memory>
#include <vector>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/expression.h"

// Forward declarations
namespace GeNN::Transpiler::Statement 
{
template<typename>
class Break;

template<typename>
class Compound;

template<typename>
class Continue;

template<typename>
class Do;

template<typename>
class Expression;

template<typename>
class For;

template<typename>
class ForEachSynapse;

template<typename>
class If;

template<typename>
class Labelled;

template<typename>
class Switch;

template<typename>
class VarDeclaration;

template<typename>
class While;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Visitor
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Statement
{
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Visitor
{
public:
    virtual void visit(const Break<A> &breakStatement) = 0;
    virtual void visit(const Compound<A> &compound) = 0;
    virtual void visit(const Continue<A> &continueStatement) = 0;
    virtual void visit(const Do<A> &doStatement) = 0;
    virtual void visit(const Expression<A> &expression) = 0;
    virtual void visit(const For<A> &forStatement) = 0;
    virtual void visit(const ForEachSynapse<A> &forEachSynapseStatement) = 0;
    virtual void visit(const If<A> &ifStatement) = 0;
    virtual void visit(const Labelled<A> &labelled) = 0;
    virtual void visit(const Switch<A> &switchStatement) = 0;
    virtual void visit(const VarDeclaration<A> &varDeclaration) = 0;
    virtual void visit(const While<A> &whileStatement) = 0;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Base
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Base
{
public:
    virtual ~Base(){}

    virtual void accept(Visitor<A> &visitor) const = 0;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Acceptable
//---------------------------------------------------------------------------
template<typename T, typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Acceptable : public Base<A>
{
public:
    virtual void accept(Visitor<A> &visitor) const final
    {
        visitor.visit(static_cast<const T&>(*this));
    }
};

template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
using StatementPtr = std::unique_ptr<Base<A> const>;

template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
using StatementList = std::vector<StatementPtr<A>>;

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Break
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Break : public Acceptable<Break<A>, A>
{
public:
    Break(Token token) 
    :   m_Token(token) 
    {}

    const auto &getToken() const { return m_Token; }

private:
    Token m_Token;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Compound
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Compound : public Acceptable<Compound<A>, A>
{
public:
    Compound(StatementList<A> statements)
    :  m_Statements(std::move(statements))
    {}

    const auto &getStatements() const { return m_Statements; }

private:
    StatementList<A> m_Statements;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Continue
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Continue : public Acceptable<Continue<A>, A>
{
public:
    Continue(Token token) 
    :   m_Token(token) 
    {}

    const auto &getToken() const { return m_Token; }

private:
    Token m_Token;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Do
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Do : public Acceptable<Do<A>, A>
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr<A>;
public:
    Do(Token whileToken, ExpressionPtr condition, StatementPtr<A> body)
    :   m_While(whileToken), m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    const auto &getWhile() const { return m_While; }
    const auto *getCondition() const { return m_Condition.get(); }
    const auto *getBody() const { return m_Body.get(); }

private:
    Token m_While;
    ExpressionPtr m_Condition;
    StatementPtr<A> m_Body;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Expression
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Expression : public Acceptable<Expression<A>, A>
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr<A>;
public:
    Expression(ExpressionPtr expression)
    :  m_Expression(std::move(expression))
    {}

    const auto *getExpression() const { return m_Expression.get(); }

private:
    ExpressionPtr m_Expression;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::For
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class For : public Acceptable<For<A>, A>
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr<A>;
public:
    For(Token forToken, StatementPtr<A> initialiser, ExpressionPtr condition, 
        ExpressionPtr increment, StatementPtr<A> body)
    :   m_For(forToken), m_Initialiser(std::move(initialiser)), m_Condition(std::move(condition)),
        m_Increment(std::move(increment)), m_Body(std::move(body))
    {}

    const auto &getFor() const { return m_For; }
    const auto *getInitialiser() const { return m_Initialiser.get(); }
    const auto *getCondition() const { return m_Condition.get(); }
    const auto *getIncrement() const { return m_Increment.get(); }
    const auto *getBody() const { return m_Body.get(); }

private:
    Token m_For;
    StatementPtr<A> m_Initialiser;
    ExpressionPtr m_Condition;
    ExpressionPtr m_Increment;
    StatementPtr<A> m_Body;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::ForEachSynapse
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class ForEachSynapse : public Acceptable<ForEachSynapse<A>, A>
{
public:
    ForEachSynapse(Token forEachSynapse, StatementPtr<A> body)
    :  m_ForEachSynapse(forEachSynapse), m_Body(std::move(body))
    {}

    const auto &getForEachSynapse() const { return m_ForEachSynapse; }
    const auto *getBody() const { return m_Body.get(); }

private:
    Token m_ForEachSynapse;
    StatementPtr<A> m_Body;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::If
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class If : public Acceptable<If<A>, A>
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr<A>;
public:
    If(Token ifToken, ExpressionPtr condition, StatementPtr<A> thenBranch, StatementPtr<A> elseBranch)
    :   m_If(ifToken), m_Condition(std::move(condition)), m_ThenBranch(std::move(thenBranch)), m_ElseBranch(std::move(elseBranch))
    {}

    const auto &getIf() const { return m_If; }
    const auto *getCondition() const { return m_Condition.get(); }
    const auto *getThenBranch() const { return m_ThenBranch.get(); }
    const auto *getElseBranch() const { return m_ElseBranch.get(); }

private:
    Token m_If;
    ExpressionPtr m_Condition;
    StatementPtr<A> m_ThenBranch;
    StatementPtr<A> m_ElseBranch;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Labelled
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Labelled : public Acceptable<Labelled<A>, A>
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr<A>;
public:
    Labelled(Token keyword, ExpressionPtr value, StatementPtr<A> body)
    :  m_Keyword(keyword), m_Value(std::move(value)), m_Body(std::move(body))
    {}

    const auto &getKeyword() const { return m_Keyword; }
    const auto *getValue() const { return m_Value.get(); }
    const auto *getBody() const { return m_Body.get(); }

private:
    Token m_Keyword;
    ExpressionPtr m_Value;
    StatementPtr<A>m_Body;
};


//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Switch
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class Switch : public Acceptable<Switch<A>, A>
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr<A>;
public:
    template<typename... AnnotationArgs>
    Switch(Token switchToken, ExpressionPtr condition, StatementPtr<A> body)
    :   m_Switch(switchToken), m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    const auto &getSwitch() const { return m_Switch; }
    const auto *getCondition() const { return m_Condition.get(); }
    const auto *getBody() const { return m_Body.get(); }
    
private:
    Token m_Switch;
    ExpressionPtr m_Condition;
    StatementPtr<A> m_Body;
};


//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::VarDeclaration
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class VarDeclaration : public Acceptable<VarDeclaration<A>, A>
{
    typedef std::vector<std::tuple<Token, GeNN::Transpiler::Expression::ExpressionPtr<A>>> InitDeclaratorList;
public:
    VarDeclaration(const Type::ResolvedType &type, InitDeclaratorList initDeclaratorList)
    :   m_Type(type), m_InitDeclaratorList(std::move(initDeclaratorList))
    {}

    const auto &getType() const{ return m_Type; }
    const auto &getInitDeclaratorList() const { return m_InitDeclaratorList; }    

private:
    Type::ResolvedType m_Type;
    std::vector<Token> m_DeclarationSpecifiers;
    InitDeclaratorList m_InitDeclaratorList;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::If
//---------------------------------------------------------------------------
template<typename A = GeNN::Transpiler::Expression::NoAnnotation>
class While : public Acceptable<While<A>, A>
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr<A>;
public:
    template<typename... AnnotationArgs>
    While(Token whileToken, ExpressionPtr condition, StatementPtr<A> body)
    :   m_While(whileToken), m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    const auto &getWhile() const{ return m_While; }
    const auto *getCondition() const { return m_Condition.get(); }
    const auto *getBody() const { return m_Body.get(); }

private:
    Token m_While;
    ExpressionPtr m_Condition;
    StatementPtr<A> m_Body;
};
}   // namespace GeNN::Transpiler::Statement
