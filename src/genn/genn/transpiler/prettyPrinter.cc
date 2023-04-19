#include "transpiler/prettyPrinter.h"

// Standard C++ includes
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <unordered_set>

// GeNN code generator includes
#include "code_generator/codeStream.h"

// Transpiler includes
#include "transpiler/transpilerUtils.h"
#include "transpiler/typeChecker.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace GeNN::Transpiler;
using namespace GeNN::Transpiler::PrettyPrinter;

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
//---------------------------------------------------------------------------
// EnvironmentInternal
//---------------------------------------------------------------------------
class EnvironmentInternal : public EnvironmentBase
{
public:
    EnvironmentInternal(EnvironmentBase &enclosing)
    :   m_Enclosing(enclosing)
    {
    }

    //---------------------------------------------------------------------------
    // EnvironmentBase virtuals
    //---------------------------------------------------------------------------
    virtual std::string define(const std::string &name) final
    {
        if(!m_LocalVariables.emplace(name).second) {
            throw std::runtime_error("Redeclaration of variable");
        }

        return "_" + name;
    }

    virtual std::string getName(const std::string &name, const Type::Base *type = nullptr) final
    {
        if(m_LocalVariables.find(name) == m_LocalVariables.end()) {
            return m_Enclosing.getName(name, type);
        }
        else {
            return "_" + name;
        }
    }

    virtual CodeStream &getStream()
    {
        return m_Enclosing.getStream();
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentBase &m_Enclosing;
    std::unordered_set<std::string> m_LocalVariables;
};

//---------------------------------------------------------------------------
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(const Statement::StatementList &statements, EnvironmentInternal &environment,
            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes)
    :   m_Environment(environment), m_Context(context), m_ResolvedTypes(resolvedTypes) 
    {
         for(auto &s : statements) {
            s.get()->accept(*this);
            m_Environment.get().getStream() << std::endl;
        }
    }

private:
    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript &arraySubscript) final
    {
        m_Environment.get().getStream() << m_Environment.get().getName(arraySubscript.getPointerName().lexeme) << "[";
        arraySubscript.getIndex()->accept(*this);
        m_Environment.get().getStream() << "]";
    }

    virtual void visit(const Expression::Assignment &assignement) final
    {
        m_Environment.get().getStream() << m_Environment.get().getName(assignement.getVarName().lexeme) << " " << assignement.getOperator().lexeme << " ";
        assignement.getValue()->accept(*this);
    }

    virtual void visit(const Expression::Binary &binary) final
    {
        binary.getLeft()->accept(*this);
        m_Environment.get().getStream() << " " << binary.getOperator().lexeme << " ";
        binary.getRight()->accept(*this);
    }

    virtual void visit(const Expression::Call &call) final
    {
        call.getCallee()->accept(*this);
        m_Environment.get().getStream() << "(";
        for(const auto &a : call.getArguments()) {
            a->accept(*this);
        }
        m_Environment.get().getStream() << ")";
    }

    virtual void visit(const Expression::Cast &cast) final
    {
        m_Environment.get().getStream() << "(";
        printType(cast.getType());
        m_Environment.get().getStream() << ")";
        cast.getExpression()->accept(*this);
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        conditional.getCondition()->accept(*this);
        m_Environment.get().getStream() << " ? ";
        conditional.getTrue()->accept(*this);
        m_Environment.get().getStream() << " : ";
        conditional.getFalse()->accept(*this);
    }

    virtual void visit(const Expression::Grouping &grouping) final
    {
        m_Environment.get().getStream() << "(";
        grouping.getExpression()->accept(*this);
        m_Environment.get().getStream() << ")";
    }

    virtual void visit(const Expression::Literal &literal) final
    {
        // If literal is a double, we want to remove the d suffix in generated code
        std::string_view lexeme = literal.getValue().lexeme;
        if (literal.getValue().type == Token::Type::DOUBLE_NUMBER){
            m_Environment.get().getStream() << lexeme.substr(0, literal.getValue().lexeme.size() - 1);
        }
        // Otherwise, if literal is a scalar, we want to add appropriate suffix for scalar type
        else if (literal.getValue().type == Token::Type::SCALAR_NUMBER) {
            const Type::NumericBase *scalar = dynamic_cast<const Type::NumericBase*>(m_Context.at("scalar"));
            m_Environment.get().getStream() << lexeme << scalar->getLiteralSuffix(m_Context);
        }
        // Otherwise, just write out original lexeme directly (strings are already quoted)
        else {
            m_Environment.get().getStream() << lexeme;
        }
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        logical.getLeft()->accept(*this);
        m_Environment.get().getStream() << " " << logical.getOperator().lexeme << " ";
        logical.getRight()->accept(*this);
    }

    virtual void visit(const Expression::PostfixIncDec &postfixIncDec) final
    {
        m_Environment.get().getStream() << m_Environment.get().getName(postfixIncDec.getVarName().lexeme) << postfixIncDec.getOperator().lexeme;
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        m_Environment.get().getStream() << prefixIncDec.getOperator().lexeme << m_Environment.get().getName(prefixIncDec.getVarName().lexeme);
    }

    virtual void visit(const Expression::Variable &variable) final
    {
        const auto *type = m_ResolvedTypes.at(&variable);
        m_Environment.get().getStream() << m_Environment.get().getName(variable.getName().lexeme, type);
    }

    virtual void visit(const Expression::Unary &unary) final
    {
        m_Environment.get().getStream() << unary.getOperator().lexeme;
        unary.getRight()->accept(*this);
    }

    //---------------------------------------------------------------------------
    // Statement::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Statement::Break&) final
    {
        m_Environment.get().getStream() << "break;";
    }

    virtual void visit(const Statement::Compound &compound) final
    {
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentInternal> oldEnvironment = m_Environment; 

        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;

        CodeGenerator::CodeStream::Scope b(m_Environment.get().getStream());
        for(auto &s : compound.getStatements()) {
            s->accept(*this);
            m_Environment.get().getStream() << std::endl;
        }

        // Restore old environment
        m_Environment = oldEnvironment;
    }

    virtual void visit(const Statement::Continue&) final
    {
        m_Environment.get().getStream() << "continue;";
    }

    virtual void visit(const Statement::Do &doStatement) final
    {
        m_Environment.get().getStream() << "do";
        doStatement.getBody()->accept(*this);
        m_Environment.get().getStream() << "while(";
        doStatement.getCondition()->accept(*this);
        m_Environment.get().getStream() << ");" << std::endl;
    }

    virtual void visit(const Statement::Expression &expression) final
    {
        expression.getExpression()->accept(*this);
        m_Environment.get().getStream() << ";";
    }

    virtual void visit(const Statement::For &forStatement) final
    {
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentInternal> oldEnvironment = m_Environment; 

        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;

        m_Environment.get().getStream() << "for(";
        if(forStatement.getInitialiser()) {
            forStatement.getInitialiser()->accept(*this);
        }
        else {
            m_Environment.get().getStream() << ";";
        }
        m_Environment.get().getStream() << " ";

        if(forStatement.getCondition()) {
            forStatement.getCondition()->accept(*this);
        }

        m_Environment.get().getStream() << "; ";
        if(forStatement.getIncrement()) {
            forStatement.getIncrement()->accept(*this);
        }
        m_Environment.get().getStream() << ")";
        forStatement.getBody()->accept(*this);

        // Restore old environment
        m_Environment = oldEnvironment;
    }

    virtual void visit(const Statement::If &ifStatement) final
    {
        m_Environment.get().getStream() << "if(";
        ifStatement.getCondition()->accept(*this);
        m_Environment.get().getStream() << ")" << std::endl;
        ifStatement.getThenBranch()->accept(*this);
        if(ifStatement.getElseBranch()) {
            m_Environment.get().getStream() << "else" << std::endl;
            ifStatement.getElseBranch()->accept(*this);
        }
    }

    virtual void visit(const Statement::Labelled &labelled) final
    {
        m_Environment.get().getStream() << labelled.getKeyword().lexeme << " ";
        if(labelled.getValue()) {
            labelled.getValue()->accept(*this);
        }
        m_Environment.get().getStream() << " : ";
        labelled.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Switch &switchStatement) final
    {
        m_Environment.get().getStream() << "switch(";
        switchStatement.getCondition()->accept(*this);
        m_Environment.get().getStream() << ")" << std::endl;
        switchStatement.getBody()->accept(*this);
    }

    virtual void visit(const Statement::VarDeclaration &varDeclaration) final
    {
        printType(varDeclaration.getType());

        for(const auto &var : varDeclaration.getInitDeclaratorList()) {
            m_Environment.get().getStream() << m_Environment.get().define(std::get<0>(var).lexeme);
            if(std::get<1>(var)) {
                m_Environment.get().getStream() << " = ";
                std::get<1>(var)->accept(*this);
            }
            m_Environment.get().getStream() << ", ";
        }
        m_Environment.get().getStream() << ";";
    }

    virtual void visit(const Statement::While &whileStatement) final
    {
        m_Environment.get().getStream() << "while(";
        whileStatement.getCondition()->accept(*this);
        m_Environment.get().getStream() << ")" << std::endl;
        whileStatement.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Print &print) final
    {
        m_Environment.get().getStream() << "print ";
        print.getExpression()->accept(*this);
        m_Environment.get().getStream() << ";";
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
        std::copy(tokens.rbegin(), tokens.rend(), std::ostream_iterator<std::string>(m_Environment.get().getStream(), " "));
    }

    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::reference_wrapper<EnvironmentInternal> m_Environment;
    const Type::TypeContext &m_Context;
    const TypeChecker::ResolvedTypeMap &m_ResolvedTypes;
};
}   // Anonymous namespace

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter
//---------------------------------------------------------------------------
void GeNN::Transpiler::PrettyPrinter::print(const Statement::StatementList &statements, EnvironmentBase &environment, 
                                            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes)
{
    EnvironmentInternal internalEnvironment(environment);
    Visitor(statements, internalEnvironment, context, resolvedTypes);
}
