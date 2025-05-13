#include "transpiler/prettyPrinter.h"

// Standard C++ includes
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stack>

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"

// Transpiler includes
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
// EnvironmentCallArgument
//---------------------------------------------------------------------------
class EnvironmentCallArgument : public EnvironmentBase
{
public:
    EnvironmentCallArgument(EnvironmentBase &enclosing)
    :   m_Enclosing(enclosing), m_CodeStream(m_Stream)
    {
    }

    //---------------------------------------------------------------------------
    // EnvironmentBase virtuals
    //---------------------------------------------------------------------------
    virtual std::string define(const std::string&) final
    {
        throw std::runtime_error("Cannot declare variable in call environment");
    }

    virtual std::string getName(const std::string &name, std::optional<Type::ResolvedType> type) final
    {
        return m_Enclosing.getName(name, type);
    }

    virtual CodeStream &getStream()
    {
        return m_CodeStream;
    }

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    std::string getString() const
    {
        return m_Stream.str(); 
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentBase &m_Enclosing;
    std::ostringstream m_Stream;
    CodeStream m_CodeStream;
};

//---------------------------------------------------------------------------
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(const Statement::StatementList &statements, EnvironmentInternal &environment,
            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes,
            StatementHandler forEachSynapseHandler)
    :   m_Environment(environment), m_Context(context), m_ResolvedTypes(resolvedTypes), m_ForEachSynapseHandler(forEachSynapseHandler)
    {
         for(auto &s : statements) {
            s.get()->accept(*this);
            m_Environment.get().getStream() << std::endl;
        }
    }

    Visitor(const Expression::ExpressionPtr &expression, EnvironmentInternal &environment,
            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes)
    :   m_Environment(environment), m_Context(context), m_ResolvedTypes(resolvedTypes) , m_ForEachSynapseHandler(nullptr)
    {
        expression.get()->accept(*this);
    }

private:
    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript &arraySubscript) final
    {
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentBase> oldEnvironment = m_Environment; 
    
        // Create new call argument environment and set to current
        EnvironmentCallArgument environment(oldEnvironment.get());
        m_Environment = environment;

        // Pretty print array index
        arraySubscript.getIndex()->accept(*this);

        // Restore old environment
        m_Environment = oldEnvironment;

        // Push arguments to top of stack
        m_CallArguments.emplace(std::piecewise_construct,
                                std::forward_as_tuple(true),
                                std::forward_as_tuple());
        m_CallArguments.top().second.push_back(environment.getString());

        // Pretty print array
        // **NOTE** like with Expression::Call, when this reaches an
        // Expression::Identifier, the indexing will get created from m_CallArguments
        arraySubscript.getArray()->accept(*this);

        // Pop stack
        m_CallArguments.pop();
    }

    virtual void visit(const Expression::Assignment &assignement) final
    {
        assignement.getAssignee()->accept(*this);
        m_Environment.get().getStream() << " " << assignement.getOperator().lexeme << " ";
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
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentBase> oldEnvironment = m_Environment; 
        
        // Loop through call arguments
        std::vector<std::string> arguments;
        arguments.reserve(call.getArguments().size());
        for (const auto &a : call.getArguments()) {
            // Create new call argument environment and set to current
            EnvironmentCallArgument environment(oldEnvironment.get());
            m_Environment = environment;

            // Pretty print argument
            a->accept(*this);
            
            // Add pretty printed argument to vector
            arguments.push_back(environment.getString());
         }

        // Restore old environment
        m_Environment = oldEnvironment;
        
        // Push arguments to top of stack
        m_CallArguments.emplace(std::piecewise_construct,
                                std::forward_as_tuple(false), 
                                std::forward_as_tuple(arguments));
        
        // Pretty print callee
        call.getCallee()->accept(*this);

        // Pop stack
        m_CallArguments.pop();

    }

    virtual void visit(const Expression::Cast &cast) final
    {
        m_Environment.get().getStream() << "(" << cast.getType().getName() << ")";
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
        // Write out lexeme
        m_Environment.get().getStream() << literal.getValue().lexeme;
        
        // If literal is a float, add f suffix
        if (literal.getValue().type == Token::Type::FLOAT_NUMBER){
            m_Environment.get().getStream() << "f";
        }
        // Otherwise, if it's an unsigned integer, add u suffix
        else if (literal.getValue().type == Token::Type::UINT32_NUMBER) {
            m_Environment.get().getStream() << "u";
        }
        // Otherwise, if literal is a scalar, return literal suffix of scalar type fro context
        else if (literal.getValue().type == Token::Type::SCALAR_NUMBER) {
            m_Environment.get().getStream() << m_Context.at("scalar").getNumeric().literalSuffix;
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
        postfixIncDec.getTarget()->accept(*this);
        m_Environment.get().getStream() <<  postfixIncDec.getOperator().lexeme;
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        m_Environment.get().getStream() << prefixIncDec.getOperator().lexeme;
        prefixIncDec.getTarget()->accept(*this);
    }

    virtual void visit(const Expression::Identifier &variable) final
    {
        // Get name of identifier
        const auto &type = m_ResolvedTypes.at(&variable);
        std::string name = m_Environment.get().getName(variable.getName().lexeme, type);

        // If identifier is function i.e. name is a function template
        if (type.isFunction()) {
            // Check that there are call arguments on the stack
            assert(!m_CallArguments.empty());

            // Loop through call arguments on top of stack
            size_t i = 0;
            for (i = 0; i < m_CallArguments.top().second.size(); i++) {
                // If name contains a $(i) placeholder to replace with this argument, replace with pretty-printed argument
                const std::string placeholder = "$(" + std::to_string(i) + ")";

                // If placeholder isn't found at all, stop looking for arguments
                size_t found = name.find(placeholder);
                if(found == std::string::npos) {
                    break;
                }
                
                // Keep replacing placeholders
                do {
                    name.replace(found, placeholder.length(), m_CallArguments.top().second.at(i));
                    found = name.find(placeholder, found);
                } while(found != std::string::npos);
            }

            // If function is variadic
            if (type.getFunction().hasFlag(Type::FunctionFlags::VARIADIC)) {
                // If variadic placeholder is found
                const std::string variadicPlaceholder = "$(@)";
                const size_t found = name.find(variadicPlaceholder);
                if (found != std::string::npos) {
                    // Concatenate together all remaining arguments
                    // **NOTE** variadic function templates shouldn't have a comma 
                    // between required and variadic arguments e.g. "printf($(0)$(@))"
                    // so, arguments simply require leading printing with leading comma
                    std::ostringstream variadicArgumentsStream;
                    const auto varArgBegin = m_CallArguments.top().second.cbegin() + i;
                    const auto varArgEnd = m_CallArguments.top().second.cend();
                    for(auto a = varArgBegin; a != varArgEnd; a++) {
                        variadicArgumentsStream << ", " << *a;
                    }
             
                    // Replace variadic placeholder with all remaining arguments 
                    name.replace(found, variadicPlaceholder.length(), variadicArgumentsStream.str());
                }
                else {
                    throw std::runtime_error("Variadic function template for '" + variable.getName().lexeme + "' (" + name + ") has "
                                             "insufficient placeholders for " + std::to_string(m_CallArguments.top().second.size()) + " argument call and no variadic placeholder '$(@)'");
                }
            }
        }
        // Otherwise, if there are array subscript arguments on top of stack
        else if(!m_CallArguments.empty() && m_CallArguments.top().first){
            assert(m_CallArguments.top().second.size() == 1);

            // Add standard indexing to name
            name += "[" + m_CallArguments.top().second.front() + "]";
        }
        
        // Print out name
        // **NOTE** this will apply any remaining substitutions
        m_Environment.get().print(name);
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
        std::reference_wrapper<EnvironmentBase> oldEnvironment = m_Environment; 

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
        if(expression.getExpression()) {
            expression.getExpression()->accept(*this);
        }
        m_Environment.get().getStream() << ";";
    }

    virtual void visit(const Statement::For &forStatement) final
    {
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentBase> oldEnvironment = m_Environment; 

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

    virtual void visit(const Statement::ForEachSynapse &forEachSynapseStatement) final
    {
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentBase> oldEnvironment = m_Environment; 

        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;

        m_ForEachSynapseHandler(m_Environment,
                                [this, &forEachSynapseStatement](EnvironmentBase &env)
                                {
                                    m_Environment = env;
                                    forEachSynapseStatement.getBody()->accept(*this);
                                });
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
        m_Environment.get().getStream() << varDeclaration.getType().getName() << " ";

        const size_t numDeclarators = varDeclaration.getInitDeclaratorList().size();
        for(size_t i = 0; i < numDeclarators; i++) {
            const auto &var = varDeclaration.getInitDeclaratorList()[i];
            m_Environment.get().getStream() << m_Environment.get().define(std::get<0>(var).lexeme);
            if(std::get<1>(var)) {
                m_Environment.get().getStream() << " = ";
                std::get<1>(var)->accept(*this);
            }
            if(i != (numDeclarators - 1)) {
                m_Environment.get().getStream() << ", ";
            }
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

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::reference_wrapper<EnvironmentBase> m_Environment;
    const Type::TypeContext &m_Context;
    const TypeChecker::ResolvedTypeMap &m_ResolvedTypes;
    StatementHandler m_ForEachSynapseHandler;
    std::stack<std::pair<bool, std::vector<std::string>>> m_CallArguments;
};
}   // Anonymous namespace

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter::EnvironmentBase
//---------------------------------------------------------------------------
void EnvironmentBase::print(const std::string &format)
{
    getStream() << printSubs(format, *this);
}
//----------------------------------------------------------------------------    
void EnvironmentBase::printLine(const std::string &format)
{
    getStream() << printSubs(format, *this) << std::endl;
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter::EnvironmentInternal
//---------------------------------------------------------------------------
std::string EnvironmentInternal::define(const std::string &name)
{
    if(!m_LocalVariables.emplace(name).second) {
        throw std::runtime_error("Redeclaration of variable");
    }

    return "_" + name;
}
//---------------------------------------------------------------------------
std::string EnvironmentInternal::getName(const std::string &name, std::optional<Type::ResolvedType> type)
{
    if(m_LocalVariables.find(name) == m_LocalVariables.end()) {
        return m_Enclosing.getName(name, type);
    }
    else {
        return "_" + name;
    }
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::PrettyPrinter
//---------------------------------------------------------------------------
void GeNN::Transpiler::PrettyPrinter::print(const Statement::StatementList &statements, EnvironmentInternal &environment, 
                                            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes,
                                            StatementHandler forEachSynapseHandler)
{
    Visitor visitor(statements, environment, context, resolvedTypes, forEachSynapseHandler);
}
//---------------------------------------------------------------------------
void GeNN::Transpiler::PrettyPrinter::print(const Expression::ExpressionPtr &expression, EnvironmentInternal &environment,
                                            const Type::TypeContext &context, const TypeChecker::ResolvedTypeMap &resolvedTypes)
{
    Visitor visitor(expression, environment, context, resolvedTypes);
}
