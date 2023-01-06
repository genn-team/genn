#pragma once

// Standard C++ includes
#include <memory>
#include <vector>

// Mini-parse includes
#include "expression.h"

// Forward declarations
namespace MiniParse::Statement 
{
class Visitor;
}
namespace Type
{
class Base;
}

//---------------------------------------------------------------------------
// MiniParse::Statement::Base
//---------------------------------------------------------------------------
namespace MiniParse::Statement
{
class Base
{
public:
    virtual void accept(Visitor &visitor) const = 0;
};

typedef std::unique_ptr<Base const> StatementPtr;
typedef std::vector<StatementPtr> StatementList;

//---------------------------------------------------------------------------
// MiniParse::Statement::Break
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
// MiniParse::Statement::Compound
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
// MiniParse::Statement::Continue
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
// MiniParse::Statement::Do
//---------------------------------------------------------------------------
class Do : public Base
{
public:
    Do(MiniParse::Expression::ExpressionPtr condition, StatementPtr body)
    :  m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const MiniParse::Expression::Base *getCondition() const { return m_Condition.get(); }
    const Base *getBody() const { return m_Body.get(); }

private:
    const MiniParse::Expression::ExpressionPtr m_Condition;
    const StatementPtr m_Body;
};

//---------------------------------------------------------------------------
// MiniParse::Statement::Expression
//---------------------------------------------------------------------------
class Expression : public Base
{
public:
    Expression(MiniParse::Expression::ExpressionPtr expression)
    :  m_Expression(std::move(expression))
    {}

    virtual void accept(Visitor &visitor) const override;

    const MiniParse::Expression::Base *getExpression() const { return m_Expression.get(); }

private:
    const MiniParse::Expression::ExpressionPtr m_Expression;
};

//---------------------------------------------------------------------------
// MiniParse::Statement::For
//---------------------------------------------------------------------------
class For : public Base
{
public:
    For(StatementPtr initialiser, MiniParse::Expression::ExpressionPtr condition, MiniParse::Expression::ExpressionPtr increment, StatementPtr body)
    :  m_Initialiser(std::move(initialiser)), m_Condition(std::move(condition)), m_Increment(std::move(increment)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const Base *getInitialiser() const { return m_Initialiser.get(); }
    const MiniParse::Expression::Base *getCondition() const { return m_Condition.get(); }
    const MiniParse::Expression::Base *getIncrement() const { return m_Increment.get(); }
    const Base *getBody() const { return m_Body.get(); }

private:
    const StatementPtr m_Initialiser;
    const MiniParse::Expression::ExpressionPtr m_Condition;
    const MiniParse::Expression::ExpressionPtr m_Increment;
    const StatementPtr m_Body;
};

//---------------------------------------------------------------------------
// MiniParse::Statement::If
//---------------------------------------------------------------------------
class If : public Base
{
public:
    If(MiniParse::Expression::ExpressionPtr condition, StatementPtr thenBranch, StatementPtr elseBranch)
    :  m_Condition(std::move(condition)), m_ThenBranch(std::move(thenBranch)), m_ElseBranch(std::move(elseBranch))
    {}

    virtual void accept(Visitor &visitor) const override;

    const MiniParse::Expression::Base *getCondition() const { return m_Condition.get(); }
    const Base *getThenBranch() const { return m_ThenBranch.get(); }
    const Base *getElseBranch() const { return m_ElseBranch.get(); }

private:
    const MiniParse::Expression::ExpressionPtr m_Condition;
    const StatementPtr m_ThenBranch;
    const StatementPtr m_ElseBranch;
};

//---------------------------------------------------------------------------
// MiniParse::Statement::Labelled
//---------------------------------------------------------------------------
class Labelled : public Base
{
public:
    Labelled(Token keyword, MiniParse::Expression::ExpressionPtr value, StatementPtr body)
    :  m_Keyword(keyword), m_Value(std::move(value)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const Token &getKeyword() const { return m_Keyword; }
    const MiniParse::Expression::Base *getValue() const { return m_Value.get(); }
    const Base *getBody() const { return m_Body.get(); }

private:
    const Token m_Keyword;
    const MiniParse::Expression::ExpressionPtr m_Value;
    const StatementPtr m_Body;
};


//---------------------------------------------------------------------------
// MiniParse::Statement::Switch
//---------------------------------------------------------------------------
class Switch : public Base
{
public:
    Switch(Token switchToken, MiniParse::Expression::ExpressionPtr condition, StatementPtr body)
    :   m_Switch(switchToken), m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const Token &getSwitch() const { return m_Switch; }
    const MiniParse::Expression::Base *getCondition() const { return m_Condition.get(); }
    const Base *getBody() const { return m_Body.get(); }
    
private:
    const Token m_Switch;
    const MiniParse::Expression::ExpressionPtr m_Condition;
    const StatementPtr m_Body;
};


//---------------------------------------------------------------------------
// MiniParse::Statement::VarDeclaration
//---------------------------------------------------------------------------
class VarDeclaration : public Base
{
public:
    typedef std::vector<std::tuple<Token, MiniParse::Expression::ExpressionPtr>> InitDeclaratorList;

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
// MiniParse::Statement::If
//---------------------------------------------------------------------------
class While : public Base
{
public:
    While(MiniParse::Expression::ExpressionPtr condition, StatementPtr body)
    :  m_Condition(std::move(condition)), m_Body(std::move(body))
    {}

    virtual void accept(Visitor &visitor) const override;

    const MiniParse::Expression::Base *getCondition() const { return m_Condition.get(); }
    const Base *getBody() const { return m_Body.get(); }

private:
    const MiniParse::Expression::ExpressionPtr m_Condition;
    const StatementPtr m_Body;
};

//---------------------------------------------------------------------------
// MiniParse::Statement::Print
//---------------------------------------------------------------------------
// **HACK** temporary until function calling is working
class Print : public Base
{
public:
    Print(MiniParse::Expression::ExpressionPtr expression)
    :  m_Expression(std::move(expression))
    {}

    virtual void accept(Visitor &visitor) const override;

    const MiniParse::Expression::Base *getExpression() const { return m_Expression.get(); }

private:
    const MiniParse::Expression::ExpressionPtr m_Expression;
};

//---------------------------------------------------------------------------
// MiniParse::Statement::Visitor
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
}   // namespace MiniParse::Statement