#include "transpiler/prettyPrinter.h"

// Standard C++ includes
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/transpilerUtils.h"

using namespace GeNN;
using namespace GeNN::Transpiler;
using namespace GeNN::Transpiler::PrettyPrinter;

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
    Visitor(const Type::TypeContext &context) : m_Context(context) {}

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
        m_StringStream << "(";
        printType(cast.getType());
        m_StringStream << ")";
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
        // If literal is a double, we want to remove the d suffix in generated code
        std::string_view lexeme = literal.getValue().lexeme;
        if (literal.getValue().type == Token::Type::DOUBLE_NUMBER){
            m_StringStream << lexeme.substr(0, literal.getValue().lexeme.size() - 1);
        }
        // Otherwise, if literal is a scalar, we want to add appropriate suffix for scalar type
        else if (literal.getValue().type == Token::Type::SCALAR_NUMBER) {
            const Type::NumericBase *scalar = dynamic_cast<const Type::NumericBase*>(m_Context.at("scalar"));
            m_StringStream << lexeme << scalar->getLiteralSuffix(m_Context);
        }
        // Otherwise, just write out original lexeme directly
        else {
            m_StringStream << lexeme;
        }
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
        printType(varDeclaration.getType());

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
    void printType(const GeNN::Type::Base *type)
    {
        // **THINK** this should be Type::getName!
        
        // Loop, building reversed list of tokens
        std::vector<std::string> tokens;
        while(true) {
            // If type is a pointer
            const auto *pointerType = dynamic_cast<const GeNN::Type::Pointer*>(type);
            if(pointerType) {
                // If pointer has const qualifier, add const
                if(pointerType->hasQualifier(GeNN::Type::Qualifier::CONSTANT)) {
                    tokens.push_back("const");
                }
                
                // Add *
                tokens.push_back("*");
                
                // Go to value type
                type = pointerType->getValueType();
            }
            // Otherwise
            else {
                // Add type specifier
                tokens.push_back(type->getName());
                
                
                if(pointerType->hasQualifier(GeNN::Type::Qualifier::CONSTANT)) {
                    tokens.push_back("const");
                }
                break;
            }
        }
        
        // Copy tokens backwards into string stream, seperating with spaces
        std::copy(tokens.rbegin(), tokens.rend(), std::ostream_iterator<std::string>(m_StringStream, " "));
        
    }
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::ostringstream m_StringStream;
    const Type::TypeContext &m_Context;
};
}   // Anonymous namespace

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter
//---------------------------------------------------------------------------
std::string GeNN::Transpiler::PrettyPrinter::print(const Statement::StatementList &statements, const Type::TypeContext &context)
{
    Visitor visitor(context);
    return visitor.print(statements);
}
