#include "pretty_printer.h"

// Standard C++ includes
#include <fstream>
#include <iostream>
#include <sstream>

// Mini-parse includes
#include "type.h"
#include "utils.h"


using namespace MiniParse;
using namespace MiniParse::PrettyPrinter;

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
//---------------------------------------------------------------------------
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    std::string print(const Statement::StatementList &statements)
    {
        // Clear string stream
        m_StringStream.str("");

        for(auto &s : statements) {
            s.get()->accept(*this);
            m_StringStream << std::endl;
        }
    
        // Return string stream contents
        return m_StringStream.str();
    }

    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript &arraySubscript) final
    {
        m_StringStream << arraySubscript.getPointerName().lexeme << "[";
        arraySubscript.getIndex()->accept(*this);
        m_StringStream << "]";
    }

    virtual void visit(const Expression::Assignment &assignement) final
    {
        m_StringStream << assignement.getVarName().lexeme << " " << assignement.getOperator().lexeme << " ";
        assignement.getValue()->accept(*this);
    }

    virtual void visit(const Expression::Binary &binary) final
    {
        binary.getLeft()->accept(*this);
        m_StringStream << " " << binary.getOperator().lexeme << " ";
        binary.getRight()->accept(*this);
    }

    virtual void visit(const Expression::Call &call) final
    {
        call.getCallee()->accept(*this);
        m_StringStream << "(";
        for(const auto &a : call.getArguments()) {
            a->accept(*this);
        }
        m_StringStream << ")";
    }

    virtual void visit(const Expression::Cast &cast) final
    {
        m_StringStream << "(" << cast.getType()->getTypeName() << ")";
        cast.getExpression()->accept(*this);
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        conditional.getCondition()->accept(*this);
        m_StringStream << " ? ";
        conditional.getTrue()->accept(*this);
        m_StringStream << " : ";
        conditional.getFalse()->accept(*this);
    }

    virtual void visit(const Expression::Grouping &grouping) final
    {
        m_StringStream << "(";
        grouping.getExpression()->accept(*this);
        m_StringStream << ")";
    }

    virtual void visit(const Expression::Literal &literal) final
    {
        std::visit(
            Utils::Overload{
                [this](auto x) { m_StringStream << x; },
                [this](std::monostate) { m_StringStream << "invalid"; }},
            literal.getValue());
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        logical.getLeft()->accept(*this);
        m_StringStream << " " << logical.getOperator().lexeme << " ";
        logical.getRight()->accept(*this);
    }

    virtual void visit(const Expression::PostfixIncDec &postfixIncDec) final
    {
        m_StringStream << postfixIncDec.getVarName().lexeme << postfixIncDec.getOperator().lexeme;
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        m_StringStream << prefixIncDec.getOperator().lexeme << prefixIncDec.getVarName().lexeme;
    }

    virtual void visit(const Expression::Variable &variable) final
    {
        m_StringStream << variable.getName().lexeme;
    }

    virtual void visit(const Expression::Unary &unary) final
    {
        m_StringStream << unary.getOperator().lexeme;
        unary.getRight()->accept(*this);
    }
    
    //---------------------------------------------------------------------------
    // Statement::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Statement::Break&) final
    {
        m_StringStream << "break;";
    }

    virtual void visit(const Statement::Compound &compound) final
    {
        m_StringStream << "{" << std::endl;
        for(auto &s : compound.getStatements()) {
            s->accept(*this);
            m_StringStream << std::endl;
        }
        m_StringStream << "}" << std::endl;
    }

    virtual void visit(const Statement::Continue&) final
    {
        m_StringStream << "continue;";
    }

    virtual void visit(const Statement::Do &doStatement) final
    {
        m_StringStream << "do";
        doStatement.getBody()->accept(*this);
        m_StringStream << "while(";
        doStatement.getCondition()->accept(*this);
        m_StringStream << ");" << std::endl;
    }

    virtual void visit(const Statement::Expression &expression) final
    {
        expression.getExpression()->accept(*this);
        m_StringStream << ";";
    }

    virtual void visit(const Statement::For &forStatement) final
    {
        m_StringStream << "for(";
        if(forStatement.getInitialiser()) {
            forStatement.getInitialiser()->accept(*this);
        }
        else {
            m_StringStream << ";";
        }
        m_StringStream << " ";

        if(forStatement.getCondition()) {
            forStatement.getCondition()->accept(*this);
        }

        m_StringStream << "; ";
        if(forStatement.getIncrement()) {
            forStatement.getIncrement()->accept(*this);
        }
        m_StringStream << ")";
        forStatement.getBody()->accept(*this);
    }

    virtual void visit(const Statement::If &ifStatement) final
    {
        m_StringStream << "if(";
        ifStatement.getCondition()->accept(*this);
        m_StringStream << ")" << std::endl;
        ifStatement.getThenBranch()->accept(*this);
        if(ifStatement.getElseBranch()) {
            m_StringStream << "else" << std::endl;
            ifStatement.getElseBranch()->accept(*this);
        }
    }

    virtual void visit(const Statement::Labelled &labelled) final
    {
        m_StringStream << labelled.getKeyword().lexeme << " ";
        if(labelled.getValue()) {
            labelled.getValue()->accept(*this);
        }
        m_StringStream << " : ";
        labelled.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Switch &switchStatement) final
    {
        m_StringStream << "switch(";
        switchStatement.getCondition()->accept(*this);
        m_StringStream << ")" << std::endl;
        switchStatement.getBody()->accept(*this);
    }

    virtual void visit(const Statement::VarDeclaration &varDeclaration) final
    {
        if(varDeclaration.isConst()) {
            m_StringStream << "const ";
        }
        m_StringStream << varDeclaration.getType()->getTypeName() << " ";

        for(const auto &var : varDeclaration.getInitDeclaratorList()) {
            m_StringStream << std::get<0>(var).lexeme;
            if(std::get<1>(var)) {
                m_StringStream << " = ";
                std::get<1>(var)->accept(*this);
            }
            m_StringStream << ", ";
        }
        m_StringStream << ";";
    }

    virtual void visit(const Statement::While &whileStatement) final
    {
        m_StringStream << "while(";
        whileStatement.getCondition()->accept(*this);
        m_StringStream << ")" << std::endl;
        whileStatement.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Print &print) final
    {
        m_StringStream << "print ";
        print.getExpression()->accept(*this);
        m_StringStream << ";";
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::ostringstream m_StringStream;
};
}   // Anonymous namespace

std::string MiniParse::PrettyPrinter::print(const Statement::StatementList &statements)
{
    Visitor visitor;
    return visitor.print(statements);
}
