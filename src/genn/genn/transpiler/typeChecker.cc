#include "transpiler/typeChecker.h"

// Standard C++ includes
#include <algorithm>
#include <string>

// Standard C includes
#include <cassert>

// GeNN includes
#include "type.h"

// Transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/expression.h"
#include "transpiler/transpilerUtils.h"

using namespace GeNN::Transpiler;
using namespace GeNN::Transpiler::TypeChecker;
namespace Type = GeNN::Type;

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
    virtual void define(const Token &name, const Type::Base *type, ErrorHandlerBase &errorHandler) final
    {
        if(!m_Types.try_emplace(name.lexeme, type).second) {
            errorHandler.error(name, "Redeclaration of variable");
            throw TypeCheckError();
        }
    }
    
    virtual const Type::Base *assign(const Token &name, Token::Type op, const Type::Base *assignedType, 
                                     const Type::TypeContext &context, ErrorHandlerBase &errorHandler, 
                                     bool initializer = false) final
    {
        // If type isn't found
        auto existingType = m_Types.find(name.lexeme);
        if(existingType == m_Types.end()) {
            return m_Enclosing.assign(name, op, assignedType,
                                      context, errorHandler, initializer);
        }
        
        // Perform standard type-checking logic
        return EnvironmentBase::assign(name, op, existingType->second, assignedType, 
                                       context, errorHandler, initializer);    
    }
    
    virtual const Type::Base *incDec(const Token &name, Token::Type op, 
                                     const Type::TypeContext &context, ErrorHandlerBase &errorHandler) final
    {
        // If type isn't found
        auto existingType = m_Types.find(name.lexeme);
        if(existingType == m_Types.end()) {
            return m_Enclosing.incDec(name, op, context, errorHandler);
        }
        
        // Perform standard type-checking logic
        return EnvironmentBase::incDec(name, op, existingType->second, errorHandler);    
    }
    
    virtual const Type::Base *getType(const Token &name, ErrorHandlerBase &errorHandler) final
    {
        auto type = m_Types.find(name.lexeme);
        if(type == m_Types.end()) {
            return m_Enclosing.getType(name, errorHandler);
        }
        else {
            return type->second;
        }
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentBase &m_Enclosing;
    std::unordered_map<std::string_view, const Type::Base*> m_Types;
};

//---------------------------------------------------------------------------
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
    :   m_Environment(nullptr), m_Type(nullptr), m_Context(context), m_ErrorHandler(errorHandler), 
        m_InLoop(false), m_InSwitch(false)
    {
    }

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    // **THINK** make constructors?
    void typeCheck(const Statement::StatementList &statements, EnvironmentInternal &environment)
    {
        m_Environment = &environment;
        for (auto &s : statements) {
            s.get()->accept(*this);
        }
    }

    const Type::Base *typeCheck(const Expression::Base *expression, EnvironmentInternal &environment)
    {
        
        m_Environment = &environment;
        return evaluateType(expression);
    }

    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript &arraySubscript) final
    {
        // Get pointer type
        auto arrayType = m_Environment->getType(arraySubscript.getPointerName(), m_ErrorHandler);
        auto pointerType = dynamic_cast<const Type::Pointer*>(arrayType);

        // If pointer is indeed a pointer
        if (pointerType) {
            // Evaluate pointer type
            auto indexType = evaluateType(arraySubscript.getIndex().get());
            auto indexNumericType = dynamic_cast<const Type::NumericBase *>(indexType);
            if (!indexNumericType || !indexNumericType->isIntegral(m_Context)) {
                m_ErrorHandler.error(arraySubscript.getPointerName(),
                                     "Invalid subscript index type '" + indexType->getName() + "'");
                throw TypeCheckError();
            }

            // Use value type of array
            m_Type = pointerType->getValueType();
        }
        // Otherwise
        else {
            m_ErrorHandler.error(arraySubscript.getPointerName(), "Subscripted object is not a pointer");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Assignment &assignment) final
    {
        const auto rhsType = evaluateType(assignment.getValue());
        m_Type = m_Environment->assign(assignment.getVarName(), assignment.getOperator().type, rhsType, 
                                       m_Context, m_ErrorHandler);
    }

    virtual void visit(const Expression::Binary &binary) final
    {
        const auto opType = binary.getOperator().type;
        const auto rightType = evaluateType(binary.getRight());
        if (opType == Token::Type::COMMA) {
            m_Type = rightType;
        }
        else {
            // If we're subtracting two pointers
            const auto leftType = evaluateType(binary.getLeft());
            auto leftNumericType = dynamic_cast<const Type::NumericBase*>(leftType);
            auto rightNumericType = dynamic_cast<const Type::NumericBase*>(rightType);
            auto leftPointerType = dynamic_cast<const Type::Pointer*>(leftType);
            auto rightPointerType = dynamic_cast<const Type::Pointer*>(rightType);
            if (leftPointerType && rightPointerType && opType == Token::Type::MINUS) {
                // Check pointers are compatible
                if (leftPointerType->getResolvedName(m_Context) != rightPointerType->getResolvedName(m_Context)) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getName() + "' and '" + rightType->getName());
                    throw TypeCheckError();
                }

                // **TODO** should be std::ptrdiff/Int64
                m_Type = Type::Int32::getInstance();
            }
            // Otherwise, if we're adding to or subtracting from pointers
            else if (leftPointerType && rightNumericType && (opType == Token::Type::PLUS || opType == Token::Type::MINUS))       // P + n or P - n
            {
                // Check that numeric operand is integer
                if (!rightNumericType->isIntegral(m_Context)) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getName() + "' and '" + rightType->getName());
                    throw TypeCheckError();
                }

                // Use left type
                m_Type = leftType;
            }
            // Otherwise, if we're adding a number to a pointer
            else if (leftNumericType && rightPointerType && opType == Token::Type::PLUS)  // n + P
            {
                // Check that numeric operand is integer
                if (!leftNumericType->isIntegral(m_Context)) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getName() + "' and '" + rightType->getName());
                    throw TypeCheckError();
                }

                // Use right type
                m_Type = leftType;
            }
            // Otherwise, if both operands are numeric
            else if (leftNumericType && rightNumericType) {
                // Otherwise, if operator requires integer operands
                if (opType == Token::Type::PERCENT || opType == Token::Type::SHIFT_LEFT
                    || opType == Token::Type::SHIFT_RIGHT || opType == Token::Type::CARET
                    || opType == Token::Type::AMPERSAND || opType == Token::Type::PIPE)
                {
                    // Check that operands are integers
                    if (!leftNumericType->isIntegral(m_Context) || !rightNumericType->isIntegral(m_Context)) {
                        m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getName() + "' and '" + rightType->getName());
                        throw TypeCheckError();
                    }

                    // If operator is a shift, promote left type
                    if (opType == Token::Type::SHIFT_LEFT || opType == Token::Type::SHIFT_RIGHT) {
                        
                        m_Type = Type::getPromotedType(leftNumericType, m_Context);
                    }
                    // Otherwise, take common type
                    else {
                        m_Type = Type::getCommonType(leftNumericType, rightNumericType, m_Context);
                    }
                }
                // Otherwise, any numeric type will do, take common type
                else {
                    m_Type = Type::getCommonType(leftNumericType, rightNumericType, m_Context);
                }
            }
            else {
                m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getName() + "' and '" + rightType->getName());
                throw TypeCheckError();
            }
        }
    }

    virtual void visit(const Expression::Call &call) final
    {
        // Evaluate callee type
        auto calleeType = evaluateType(call.getCallee());
        auto calleeFunctionType = dynamic_cast<const Type::ForeignFunctionBase *>(calleeType);

        // If callee's a function
        if (calleeFunctionType) {
            // If argument count doesn't match
            const auto argTypes = calleeFunctionType->getArgumentTypes();
            if (call.getArguments().size() < argTypes.size()) {
                m_ErrorHandler.error(call.getClosingParen(), "Too many arguments to function");
                throw TypeCheckError();
            }
            else if (call.getArguments().size() > argTypes.size()) {
                m_ErrorHandler.error(call.getClosingParen(), "Too few arguments to function");
                throw TypeCheckError();
            }
            else {
                // Loop through arguments
                // **TODO** check
                /*for(size_t i = 0; i < argTypes.size(); i++) {
                    // Evaluate argument type
                    auto callArgType = evaluateType(call.getArguments().at(i).get());
                }*/
                // Type is return type of function
                m_Type = calleeFunctionType->getReturnType();
            }
        }
        // Otherwise
        else {
            m_ErrorHandler.error(call.getClosingParen(), "Called object is not a function");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Cast &cast) final
    {
        // Evaluate type of expression we're casting
        const auto rightType = evaluateType(cast.getExpression());
        
        // If const is being removed
        if (rightType->hasQualifier(Type::Qualifier::CONSTANT) && !cast.getType()->hasQualifier(Type::Qualifier::CONSTANT)) {
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getType()->getName() + "' and '" + rightType->getName());
            throw TypeCheckError();
        }

        // If we're trying to cast pointer to pointer
        auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType);
        auto rightPointerType = dynamic_cast<const Type::Pointer *>(rightType);
        auto leftNumericType = dynamic_cast<const Type::NumericBase *>(cast.getType());
        auto leftPointerType = dynamic_cast<const Type::Pointer *>(cast.getType());
        if (rightPointerType && leftPointerType) {
            if (rightPointerType->getResolvedName(m_Context) != leftPointerType->getResolvedName(m_Context)) {
                m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getType()->getName() + "' and '" + rightType->getName());
                throw TypeCheckError();
            }
        }
        // Otherwise, if either operand isn't numeric
        else if(!leftNumericType | !rightNumericType) {
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getType()->getName() + "' and '" + rightType->getName());
            throw TypeCheckError();
        }

        m_Type = cast.getType();
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        const auto trueType = evaluateType(conditional.getTrue());
        const auto falseType = evaluateType(conditional.getFalse());
        auto trueNumericType = dynamic_cast<const Type::NumericBase *>(trueType);
        auto falseNumericType = dynamic_cast<const Type::NumericBase *>(falseType);
        if (trueNumericType && falseNumericType) {
            // **TODO** check behaviour
            m_Type = Type::getCommonType(trueNumericType, falseNumericType, m_Context);
            if(trueType->hasQualifier(Type::Qualifier::CONSTANT) || falseType->hasQualifier(Type::Qualifier::CONSTANT)) {
                m_Type = m_Type->getQualifiedType(Type::Qualifier::CONSTANT);
            }
        }
        else {
            m_ErrorHandler.error(conditional.getQuestion(),
                                 "Invalid operand types '" + trueType->getName() + "' and '" + falseType->getName() + "' to conditional");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Grouping &grouping) final
    {
        m_Type = evaluateType(grouping.getExpression());
    }

    virtual void visit(const Expression::Literal &literal) final
    {
        // Convert number token type to type
        // **THINK** is it better to use typedef for scalar or resolve from m_Context
        if (literal.getValue().type == Token::Type::DOUBLE_NUMBER) {
            m_Type = Type::Double::getInstance();
        }
        else if (literal.getValue().type == Token::Type::FLOAT_NUMBER) {
            m_Type = Type::Double::getInstance();
        }
        else if (literal.getValue().type == Token::Type::SCALAR_NUMBER) {
            // **TODO** cache
            m_Type = new Type::NumericTypedef("scalar");
        }
        else if (literal.getValue().type == Token::Type::INT32_NUMBER) {
            m_Type = Type::Int32::getInstance();
        }
        else if (literal.getValue().type == Token::Type::UINT32_NUMBER) {
            m_Type = Type::Uint32::getInstance();
        }
        else {
            assert(false);
        }
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        logical.getLeft()->accept(*this);
        logical.getRight()->accept(*this);
        m_Type = Type::Int32::getInstance();
    }

    virtual void visit(const Expression::PostfixIncDec &postfixIncDec) final
    {
        m_Type = m_Environment->incDec(postfixIncDec.getVarName(), postfixIncDec.getOperator().type, 
                                       m_Context, m_ErrorHandler);
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        m_Type = m_Environment->incDec(prefixIncDec.getVarName(), prefixIncDec.getOperator().type, 
                                       m_Context, m_ErrorHandler);
    }

    virtual void visit(const Expression::Variable &variable)
    {
        m_Type = m_Environment->getType(variable.getName(), m_ErrorHandler);
    }

    virtual void visit(const Expression::Unary &unary) final
    {
        const auto rightType = evaluateType(unary.getRight());

        // If operator is pointer de-reference
        if (unary.getOperator().type == Token::Type::STAR) {
            auto rightPointerType = dynamic_cast<const Type::Pointer *>(rightType);
            if (!rightPointerType) {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType->getName() + "'");
                throw TypeCheckError();
            }

            // Return value type
            m_Type = rightPointerType->getValueType();
        }
        // Otherwise
        else {
            auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType);
            if (rightNumericType) {
                // If operator is arithmetic, return promoted type
                if (unary.getOperator().type == Token::Type::PLUS || unary.getOperator().type == Token::Type::MINUS) {
                    // **THINK** const through these?
                    m_Type = Type::getPromotedType(rightNumericType, m_Context);
                }
                // Otherwise, if operator is bitwise
                else if (unary.getOperator().type == Token::Type::TILDA) {
                    // If type is integer, return promoted type
                    if (rightNumericType->isIntegral(m_Context)) {
                        // **THINK** const through these?
                        m_Type = Type::getPromotedType(rightNumericType, m_Context);
                    }
                    else {
                        m_ErrorHandler.error(unary.getOperator(),
                                             "Invalid operand type '" + rightType->getName() + "'");
                        throw TypeCheckError();
                    }
                }
                // Otherwise, if operator is logical
                else if (unary.getOperator().type == Token::Type::NOT) {
                    m_Type = Type::Int32::getInstance();;
                }
                // Otherwise, if operator is address of, return pointer type
                else if (unary.getOperator().type == Token::Type::AMPERSAND) {
                    m_Type = rightType->getPointerType();
                }
            }
            else {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType->getName() + "'");
                throw TypeCheckError();
            }
        }
    }

    //---------------------------------------------------------------------------
    // Statement::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Statement::Break &breakStatement) final
    {
        if (!m_InLoop && !m_InSwitch) {
            m_ErrorHandler.error(breakStatement.getToken(), "Statement not within loop");
        }
    }

    virtual void visit(const Statement::Compound &compound) final
    {
        EnvironmentInternal environment(*m_Environment);
        typeCheck(compound.getStatements(), environment);
    }

    virtual void visit(const Statement::Continue &continueStatement) final
    {
        if (!m_InLoop) {
            m_ErrorHandler.error(continueStatement.getToken(), "Statement not within loop");
        }
    }

    virtual void visit(const Statement::Do &doStatement) final
    {
        m_InLoop = true;
        doStatement.getBody()->accept(*this);
        m_InLoop = false;
        doStatement.getCondition()->accept(*this);
    }

    virtual void visit(const Statement::Expression &expression) final
    {
        expression.getExpression()->accept(*this);
    }

    virtual void visit(const Statement::For &forStatement) final
    {
        // Create new environment for loop initialisation
        EnvironmentInternal *previous = m_Environment;
        EnvironmentInternal environment(*m_Environment);
        m_Environment = &environment;

        // Interpret initialiser if statement present
        if (forStatement.getInitialiser()) {
            forStatement.getInitialiser()->accept(*this);
        }

        if (forStatement.getCondition()) {
            forStatement.getCondition()->accept(*this);
        }

        if (forStatement.getIncrement()) {
            forStatement.getIncrement()->accept(*this);
        }

        m_InLoop = true;
        forStatement.getBody()->accept(*this);
        m_InLoop = false;

        // Restore environment
        m_Environment = previous;
    }

    virtual void visit(const Statement::If &ifStatement) final
    {
        ifStatement.getCondition()->accept(*this);
        ifStatement.getThenBranch()->accept(*this);
        if (ifStatement.getElseBranch()) {
            ifStatement.getElseBranch()->accept(*this);
        }
    }

    virtual void visit(const Statement::Labelled &labelled) final
    {
        if (!m_InSwitch) {
            m_ErrorHandler.error(labelled.getKeyword(), "Statement not within switch statement");
        }

        if (labelled.getValue()) {
            auto valType = evaluateType(labelled.getValue());
            auto valNumericType = dynamic_cast<const Type::NumericBase *>(valType);
            if (!valNumericType || !valNumericType->isIntegral(m_Context)) {
                m_ErrorHandler.error(labelled.getKeyword(),
                                     "Invalid case value '" + valType->getName() + "'");
                throw TypeCheckError();
            }
        }

        labelled.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Switch &switchStatement) final
    {
        auto condType = evaluateType(switchStatement.getCondition());
        auto condNumericType = dynamic_cast<const Type::NumericBase *>(condType);
        if (!condNumericType || !condNumericType->isIntegral(m_Context)) {
            m_ErrorHandler.error(switchStatement.getSwitch(),
                                 "Invalid condition '" + condType->getName() + "'");
            throw TypeCheckError();
        }

        m_InSwitch = true;
        switchStatement.getBody()->accept(*this);
        m_InSwitch = false;
    }

    virtual void visit(const Statement::VarDeclaration &varDeclaration) final
    {
        for (const auto &var : varDeclaration.getInitDeclaratorList()) {
            m_Environment->define(std::get<0>(var), varDeclaration.getType(), m_ErrorHandler);

            // If variable has an initialiser expression
            if (std::get<1>(var)) {
                // Evaluate type
                const auto initialiserType = evaluateType(std::get<1>(var).get());

                // Assign initialiser expression to variable
                m_Environment->assign(std::get<0>(var), Token::Type::EQUAL, initialiserType, 
                                      m_Context, m_ErrorHandler, true);
            }
        }
    }

    virtual void visit(const Statement::While &whileStatement) final
    {
        whileStatement.getCondition()->accept(*this);
        m_InLoop = true;
        whileStatement.getBody()->accept(*this);
        m_InLoop = false;
    }

    virtual void visit(const Statement::Print &print) final
    {
        print.getExpression()->accept(*this);
    }

private:
    //---------------------------------------------------------------------------
    // Private methods
    //---------------------------------------------------------------------------
    const Type::Base *evaluateType(const Expression::Base *expression)
    {
        expression->accept(*this);
        return m_Type;
    }

    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentInternal *m_Environment;
    const Type::Base *m_Type;
    const Type::TypeContext &m_Context;
    ErrorHandlerBase &m_ErrorHandler;
    bool m_InLoop;
    bool m_InSwitch;
};
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentBase
//---------------------------------------------------------------------------
const Type::Base *EnvironmentBase::assign(const Token &name, Token::Type op, 
                                          const Type::Base *existingType, const Type::Base *assignedType, 
                                          const Type::TypeContext &context, ErrorHandlerBase &errorHandler, 
                                          bool initializer) const
{
    // If existing type is a const qualified and isn't being initialized, give error
    if(!initializer && existingType->hasQualifier(Type::Qualifier::CONSTANT)) {
        errorHandler.error(name, "Assignment of read-only variable");
        throw TypeCheckError();
    }
    
    // If assignment operation is plain equals, any type is fine so return
    auto numericExistingType = dynamic_cast<const Type::NumericBase *>(existingType);
    auto pointerExistingType = dynamic_cast<const Type::Pointer *>(existingType);
    auto numericAssignedType = dynamic_cast<const Type::NumericBase *>(assignedType);
    auto pointerAssignedType = dynamic_cast<const Type::Pointer *>(assignedType);
    if(op == Token::Type::EQUAL) {
        // If we're initialising a pointer with another pointer
        if (pointerAssignedType && pointerExistingType) {
            // If we're trying to assign a pointer to a const value to a pointer
            if (assignedType->hasQualifier(Type::Qualifier::CONSTANT) && !existingType->hasQualifier(Type::Qualifier::CONSTANT)) {
                errorHandler.error(name, "Invalid operand types '" + pointerExistingType->getName() + "' and '" + pointerAssignedType->getName());
                throw TypeCheckError();
            }

            // If pointer types aren't compatible
            if (pointerExistingType->getResolvedName(context) != pointerAssignedType->getResolvedName(context)) {
                errorHandler.error(name, "Invalid operand types '" + pointerExistingType->getName() + "' and '" + pointerAssignedType->getName());
                throw TypeCheckError();
            }
        }
        // Otherwise, if we're trying to initialise a pointer with a non-pointer or vice-versa
        else if (pointerAssignedType || pointerExistingType) {
            errorHandler.error(name, "Invalid operand types '" + existingType->getName() + "' and '" + assignedType->getName());
            throw TypeCheckError();
        }
    }
    // Otherwise, if operation is += or --
    else if (op == Token::Type::PLUS_EQUAL || op == Token::Type::MINUS_EQUAL) {
        // If the operand being added isn't numeric or the type being added to is neither numeric or a pointer
        if (!numericAssignedType || (!pointerExistingType && !numericExistingType))
        {
            errorHandler.error(name, "Invalid operand types '" + existingType->getName() + "' and '" + assignedType->getName() + "'");
            throw TypeCheckError();
        }

        // If we're adding a numeric type to a pointer, check it's an integer
        if (pointerExistingType && numericAssignedType->isIntegral(context)) {
            errorHandler.error(name, "Invalid operand types '" + numericAssignedType->getName() + "'");
            throw TypeCheckError();
        }
    }
    // Otherwise, numeric types are required
    else {
        // If either type is non-numeric, give error
        if(!numericAssignedType) {
            errorHandler.error(name, "Invalid operand types '" + numericAssignedType->getName() + "'");
            throw TypeCheckError();
        }
        if(!numericExistingType) {
            errorHandler.error(name, "Invalid operand types '" + existingType->getName() + "'");
            throw TypeCheckError();
        }

        // If operand isn't one that takes any numeric type, check both operands are integral
        if (op != Token::Type::STAR_EQUAL && op != Token::Type::SLASH_EQUAL) {
            if(!numericAssignedType->isIntegral(context)) {
                errorHandler.error(name, "Invalid operand types '" + numericAssignedType->getName() + "'");
                throw TypeCheckError();
            }
            if(!numericExistingType->isIntegral(context)) {
                errorHandler.error(name, "Invalid operand types '" + numericExistingType->getName() + "'");
                throw TypeCheckError();
            }
        }
    }
   
     // Return existing type
     // **THINK**
    return existingType;
}
//---------------------------------------------------------------------------
const Type::Base *EnvironmentBase::incDec(const Token &name, Token::Type, 
                                          const Type::Base *existingType, ErrorHandlerBase &errorHandler) const
{
    // If existing type has a constant qualifier, give errors
    if(existingType->hasQualifier(Type::Qualifier::CONSTANT)) {
        errorHandler.error(name, "Increment/decrement of read-only variable");
        throw TypeCheckError();
    }
    // Otherwise, return type
    else {
        return existingType;
    }
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker
//---------------------------------------------------------------------------
void GeNN::Transpiler::TypeChecker::typeCheck(const Statement::StatementList &statements, EnvironmentBase &environment, 
                                              const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    Visitor visitor(context, errorHandler);
    EnvironmentInternal internalEnvironment(environment);
    visitor.typeCheck(statements, internalEnvironment);
}
//---------------------------------------------------------------------------
const Type::Base *GeNN::Transpiler::TypeChecker::typeCheck(const Expression::Base *expression, EnvironmentBase &environment,
                                                           const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    Visitor visitor(context, errorHandler);
    EnvironmentInternal internalEnvironment(environment);
    return visitor.typeCheck(expression, internalEnvironment);
}
