#pragma once

// Standard C++ includes
#include <memory>
#include <vector>

// Transpiler includes
#include "transpiler/expression.h"

// Forward declarations
namespace GeNN::Transpiler::Statement 
{
class Visitor;
}
namespace GeNN::Type
{
class Base;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Base
//---------------------------------------------------------------------------
namespace GeNN::Transpiler::Statement
{
class Base
{
public:
    virtual void accept(Visitor &visitor) const = 0;
};

typedef std::unique_ptr<Base const> StatementPtr;
typedef std::vector<StatementPtr> StatementList;

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Break
//---------------------------------------------------------------------------
class Break : public Base
{
public:
    Break(Token token) 
    :   m_Token(token) 
    {}

    virtual void accept(Visitor &visitor) const override;

    const Token &getToken() const { return m_Token; }

private:
    const Token m_Token;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Compound
//---------------------------------------------------------------------------
class Compound : public Base
{
public:
    Compound(StatementList statements)
    :  m_Statements(std::move(statements))
    {}

    virtual void accept(Visitor &visitor) const override;

    const StatementList &getStatements() const { return m_Statements; }

private:
    const StatementList m_Statements;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Continue
//---------------------------------------------------------------------------
class Continue : public Base
{
public:
    Continue(Token token) 
    :   m_Token(token) 
    {}

    virtual void accept(Visitor &visitor) const override;

    const Token &getToken() const { return m_Token; }

private:
    const Token m_Token;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Do
//---------------------------------------------------------------------------
class Do : public Base
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr;
public:
    Do(ExpressionPtr condition, StatementPtr body)
    :  m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const ExpressionPtr::element_type *getCondition() const { return m_Condition.get(); }
    const Base *getBody() const { return m_Body.get(); }

private:
    const ExpressionPtr m_Condition;
    const StatementPtr m_Body;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Expression
//---------------------------------------------------------------------------
class Expression : public Base
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr;
public:
    Expression(ExpressionPtr expression)
    :  m_Expression(std::move(expression))
    {}

    virtual void accept(Visitor &visitor) const override;

    const ExpressionPtr::element_type *getExpression() const { return m_Expression.get(); }

private:
    const ExpressionPtr m_Expression;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::For
//---------------------------------------------------------------------------
class For : public Base
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr;
public:
    For(StatementPtr initialiser, ExpressionPtr condition, ExpressionPtr increment, StatementPtr body)
    :  m_Initialiser(std::move(initialiser)), m_Condition(std::move(condition)), m_Increment(std::move(increment)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const Base *getInitialiser() const { return m_Initialiser.get(); }
    const ExpressionPtr::element_type *getCondition() const { return m_Condition.get(); }
    const ExpressionPtr::element_type *getIncrement() const { return m_Increment.get(); }
    const Base *getBody() const { return m_Body.get(); }

private:
    const StatementPtr m_Initialiser;
    const ExpressionPtr m_Condition;
    const ExpressionPtr m_Increment;
    const StatementPtr m_Body;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::If
//---------------------------------------------------------------------------
class If : public Base
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr;
public:
    If(ExpressionPtr condition, StatementPtr thenBranch, StatementPtr elseBranch)
    :  m_Condition(std::move(condition)), m_ThenBranch(std::move(thenBranch)), m_ElseBranch(std::move(elseBranch))
    {}

    virtual void accept(Visitor &visitor) const override;

    const ExpressionPtr::element_type *getCondition() const { return m_Condition.get(); }
    const Base *getThenBranch() const { return m_ThenBranch.get(); }
    const Base *getElseBranch() const { return m_ElseBranch.get(); }

private:
    const ExpressionPtr m_Condition;
    const StatementPtr m_ThenBranch;
    const StatementPtr m_ElseBranch;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Labelled
//---------------------------------------------------------------------------
class Labelled : public Base
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr;
public:
    Labelled(Token keyword, ExpressionPtr value, StatementPtr body)
    :  m_Keyword(keyword), m_Value(std::move(value)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const Token &getKeyword() const { return m_Keyword; }
    const ExpressionPtr::element_type *getValue() const { return m_Value.get(); }
    const Base *getBody() const { return m_Body.get(); }

private:
    const Token m_Keyword;
    const ExpressionPtr m_Value;
    const StatementPtr m_Body;
};


//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Switch
//---------------------------------------------------------------------------
class Switch : public Base
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr;
public:
    Switch(Token switchToken, ExpressionPtr condition, StatementPtr body)
    :   m_Switch(switchToken), m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const Token &getSwitch() const { return m_Switch; }
    const ExpressionPtr::element_type *getCondition() const { return m_Condition.get(); }
    const Base *getBody() const { return m_Body.get(); }
    
private:
    const Token m_Switch;
    const ExpressionPtr m_Condition;
    const StatementPtr m_Body;
};


//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::VarDeclaration
//---------------------------------------------------------------------------
class VarDeclaration : public Base
{
public:
    typedef std::vector<std::tuple<Token, GeNN::Transpiler::Expression::ExpressionPtr>> InitDeclaratorList;

    VarDeclaration(const Type::Base *type, bool isConst, InitDeclaratorList initDeclaratorList)
    :   m_Type(type), m_Const(isConst), m_InitDeclaratorList(std::move(initDeclaratorList))
    {}

    virtual void accept(Visitor &visitor) const override;

    const Type::Base *getType() const { return m_Type; }
    bool isConst() const { return m_Const; }
    
    const InitDeclaratorList &getInitDeclaratorList() const { return m_InitDeclaratorList; }
    
private:
    const Type::Base *m_Type;
    const bool m_Const;
    const std::vector<Token> m_DeclarationSpecifiers;
    const InitDeclaratorList m_InitDeclaratorList;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::If
//---------------------------------------------------------------------------
class While : public Base
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr;
public:
    While(ExpressionPtr condition, StatementPtr body)
    :  m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const ExpressionPtr::element_type *getCondition() const { return m_Condition.get(); }
    const Base *getBody() const { return m_Body.get(); }

private:
    const ExpressionPtr m_Condition;
    const StatementPtr m_Body;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Print
//---------------------------------------------------------------------------
// **HACK** temporary until function calling is working
class Print : public Base
{
    using ExpressionPtr = GeNN::Transpiler::Expression::ExpressionPtr;
public:
    Print(ExpressionPtr expression)
    :  m_Expression(std::move(expression))
    {}

    virtual void accept(Visitor &visitor) const override;

    const ExpressionPtr::element_type *getExpression() const { return m_Expression.get(); }

private:
    const ExpressionPtr m_Expression;
};

//---------------------------------------------------------------------------
// GeNN::Transpiler::Statement::Visitor
//---------------------------------------------------------------------------
class Visitor
{
public:
    virtual void visit(const Break &breakStatement) = 0;
    virtual void visit(const Compound &compound) = 0;
    virtual void visit(const Continue &continueStatement) = 0;
    virtual void visit(const Do &doStatement) = 0;
    virtual void visit(const Expression &expression) = 0;
    virtual void visit(const For &forStatement) = 0;
    virtual void visit(const If &ifStatement) = 0;
    virtual void visit(const Labelled &labelled) = 0;
    virtual void visit(const Switch &switchStatement) = 0;
    virtual void visit(const VarDeclaration &varDeclaration) = 0;
    virtual void visit(const While &whileStatement) = 0;
    virtual void visit(const Print &print) = 0;
};
}   // namespace GeNN::Transpiler::Statement