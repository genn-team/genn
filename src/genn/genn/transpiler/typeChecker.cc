#include "transpiler/typeChecker.h"

// Standard C++ includes
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
    virtual void define(const Token &name, const Type::QualifiedType &qualifiedType, ErrorHandlerBase &errorHandler) final
    {
        if(!m_Types.try_emplace(name.lexeme, qualifiedType).second) {
            errorHandler.error(name, "Redeclaration of variable");
            throw TypeCheckError();
        }
    }
    
    virtual const Type::QualifiedType &assign(const Token &name, Token::Type op, const Type::QualifiedType &assignedType, 
                                              ErrorHandlerBase &errorHandler, bool initializer = false) final
    {
        // If type isn't found
        auto existingType = m_Types.find(name.lexeme);
        if(existingType == m_Types.end()) {
            return m_Enclosing.assign(name, op, assignedType,
                                      errorHandler, initializer);
        }
        
        // Perform standard type-checking logic
        return EnvironmentBase::assign(name, op, existingType->second, assignedType, errorHandler, initializer);    
    }
    
    virtual const Type::QualifiedType &incDec(const Token &name, Token::Type op, ErrorHandlerBase &errorHandler) final
    {
        // If type isn't found
        auto existingType = m_Types.find(name.lexeme);
        if(existingType == m_Types.end()) {
            return m_Enclosing.incDec(name, op, errorHandler);
        }
        
        // Perform standard type-checking logic
        return EnvironmentBase::incDec(name, op, existingType->second, errorHandler);    
    }
    
    virtual const Type::QualifiedType &getType(const Token &name, ErrorHandlerBase &errorHandler) final
    {
        auto type = m_Types.find(std::string{name.lexeme});
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
    std::unordered_map<std::string_view, Type::QualifiedType> m_Types;
};

//---------------------------------------------------------------------------
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(ErrorHandlerBase &errorHandler)
    :   m_Environment(nullptr), m_QualifiedType{nullptr, false, false}, m_ErrorHandler(errorHandler), 
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

    const Type::QualifiedType typeCheck(const Expression::Base *expression, EnvironmentInternal &environment)
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
        auto pointerType = dynamic_cast<const Type::Pointer*>(arrayType.type);

        // If pointer is indeed a pointer
        if (pointerType) {
            // Evaluate pointer type
            auto indexType = evaluateType(arraySubscript.getIndex().get());
            auto indexNumericType = dynamic_cast<const Type::NumericBase *>(indexType.type);
            if (!indexNumericType || !indexNumericType->isIntegral()) {
                m_ErrorHandler.error(arraySubscript.getPointerName(),
                                     "Invalid subscript index type '" + indexType.type->getTypeName() + "'");
                throw TypeCheckError();
            }

            // Use value type of array
            m_QualifiedType = Type::QualifiedType{pointerType->getValueType(), arrayType.constValue, false};
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
        m_QualifiedType = m_Environment->assign(assignment.getVarName(), assignment.getOperator().type, rhsType, m_ErrorHandler);
    }

    virtual void visit(const Expression::Binary &binary) final
    {
        const auto opType = binary.getOperator().type;
        const auto rightType = evaluateType(binary.getRight());
        if (opType == Token::Type::COMMA) {
            m_QualifiedType = rightType;
        }
        else {
            // If we're subtracting two pointers
            const auto leftType = evaluateType(binary.getLeft());
            auto leftNumericType = dynamic_cast<const Type::NumericBase *>(leftType.type);
            auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType.type);
            auto leftPointerType = dynamic_cast<const Type::Pointer *>(leftType.type);
            auto rightPointerType = dynamic_cast<const Type::Pointer *>(rightType.type);
            if (leftPointerType && rightPointerType && opType == Token::Type::MINUS) {
                // Check pointers are compatible
                if (leftPointerType->getTypeName() != rightPointerType->getTypeName()) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType.type->getTypeName() + "' and '" + rightType.type->getTypeName());
                    throw TypeCheckError();
                }

                // **TODO** should be std::ptrdiff/Int64
                m_QualifiedType = Type::QualifiedType{Type::Int32::getInstance(), false, false};
            }
            // Otherwise, if we're adding to or subtracting from pointers
            else if (leftPointerType && rightNumericType && (opType == Token::Type::PLUS || opType == Token::Type::MINUS))       // P + n or P - n
            {
                // Check that numeric operand is integer
                if (!rightNumericType->isIntegral()) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType.type->getTypeName() + "' and '" + rightType.type->getTypeName());
                    throw TypeCheckError();
                }

                // Use left type
                m_QualifiedType = leftType;
            }
            // Otherwise, if we're adding a number to a pointer
            else if (leftNumericType && rightPointerType && opType == Token::Type::PLUS)  // n + P
            {
                // Check that numeric operand is integer
                if (!leftNumericType->isIntegral()) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType.type->getTypeName() + "' and '" + rightType.type->getTypeName());
                    throw TypeCheckError();
                }

                // Use right type
                m_QualifiedType = leftType;
            }
            // Otherwise, if both operands are numeric
            else if (leftNumericType && rightNumericType) {
                // Otherwise, if operator requires integer operands
                if (opType == Token::Type::PERCENT || opType == Token::Type::SHIFT_LEFT
                    || opType == Token::Type::SHIFT_RIGHT || opType == Token::Type::CARET
                    || opType == Token::Type::AMPERSAND || opType == Token::Type::PIPE)
                {
                    // Check that operands are integers
                    if (!leftNumericType->isIntegral() || !rightNumericType->isIntegral()) {
                        m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType.type->getTypeName() + "' and '" + rightType.type->getTypeName());
                        throw TypeCheckError();
                    }

                    // If operator is a shift, promote left type
                    if (opType == Token::Type::SHIFT_LEFT || opType == Token::Type::SHIFT_RIGHT) {
                        
                        m_QualifiedType = Type::QualifiedType{Type::getPromotedType(leftNumericType), false, false};
                    }
                    // Otherwise, take common type
                    else {
                        m_QualifiedType = Type::QualifiedType{Type::getCommonType(leftNumericType, rightNumericType), false, false};
                    }
                }
                // Otherwise, any numeric type will do, take common type
                else {
                    m_QualifiedType = Type::QualifiedType{Type::getCommonType(leftNumericType, rightNumericType), false, false};
                }
            }
            else {
                m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType.type->getTypeName() + "' and '" + rightType.type->getTypeName());
                throw TypeCheckError();
            }
        }
    }

    virtual void visit(const Expression::Call &call) final
    {
        // Evaluate callee type
        auto calleeType = evaluateType(call.getCallee());
        auto calleeFunctionType = dynamic_cast<const Type::ForeignFunctionBase *>(calleeType.type);

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
                m_QualifiedType = Type::QualifiedType{calleeFunctionType->getReturnType(), false, false};
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
        
        // If value const is being removed
        if (rightType.constValue && !cast.getQualifiedType().constValue) {
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getQualifiedType().type->getTypeName() + "' and '" + rightType.type->getTypeName());
            throw TypeCheckError();
        }
        // Otherwise, if pointer const is being removed
        else if (rightType.constPointer && !cast.getQualifiedType().constPointer) {
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getQualifiedType().type->getTypeName() + "' and '" + rightType.type->getTypeName());
            throw TypeCheckError();
        }

        // If we're trying to cast pointer to pointer
        auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType.type);
        auto rightPointerType = dynamic_cast<const Type::Pointer *>(rightType.type);
        auto leftNumericType = dynamic_cast<const Type::NumericBase *>(cast.getQualifiedType().type);
        auto leftPointerType = dynamic_cast<const Type::Pointer *>(cast.getQualifiedType().type);
        if (rightPointerType && leftPointerType) {
            if (rightPointerType->getTypeName() != leftPointerType->getTypeName()) {
                m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getQualifiedType().type->getTypeName() + "' and '" + rightType.type->getTypeName());
                throw TypeCheckError();
            }
        }
        // Otherwise, if either operand isn't numeric
        else if(!leftNumericType | !rightNumericType) {
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getQualifiedType().type->getTypeName() + "' and '" + rightType.type->getTypeName());
            throw TypeCheckError();
        }

        m_QualifiedType = cast.getQualifiedType();
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        const auto trueType = evaluateType(conditional.getTrue());
        const auto falseType = evaluateType(conditional.getFalse());
        auto trueNumericType = dynamic_cast<const Type::NumericBase *>(trueType.type);
        auto falseNumericType = dynamic_cast<const Type::NumericBase *>(falseType.type);
        if (trueNumericType && falseNumericType) {
            // **TODO** check behaviour
            m_QualifiedType = Type::QualifiedType{Type::getCommonType(trueNumericType, falseNumericType),
                                                  trueType.constValue || falseType.constValue,
                                                  trueType.constPointer || falseType.constPointer};
        }
        else {
            m_ErrorHandler.error(conditional.getQuestion(),
                                 "Invalid operand types '" + trueType.type->getTypeName() + "' and '" + falseType.type->getTypeName() + "' to conditional");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Grouping &grouping) final
    {
        m_QualifiedType = evaluateType(grouping.getExpression());
    }

    virtual void visit(const Expression::Literal &literal) final
    {
        m_QualifiedType = Type::QualifiedType{
            std::visit(Utils::Overload{
                           [](auto v)->const Type::NumericBase *{ return Type::TypeTraits<decltype(v)>::NumericType::getInstance(); },
                           [](std::monostate)->const Type::NumericBase *{ return nullptr; }},
                           literal.getValue()),
            true,
            false};
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        logical.getLeft()->accept(*this);
        logical.getRight()->accept(*this);
        m_QualifiedType = Type::QualifiedType{Type::Int32::getInstance(), false, false};
    }

    virtual void visit(const Expression::PostfixIncDec &postfixIncDec) final
    {
        m_QualifiedType = m_Environment->incDec(postfixIncDec.getVarName(),
                                                postfixIncDec.getOperator().type, m_ErrorHandler);
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        m_QualifiedType = m_Environment->incDec(prefixIncDec.getVarName(),
                                                prefixIncDec.getOperator().type, m_ErrorHandler);
    }

    virtual void visit(const Expression::Variable &variable)
    {
        m_QualifiedType = m_Environment->getType(variable.getName(), m_ErrorHandler);
    }

    virtual void visit(const Expression::Unary &unary) final
    {
        const auto rightType = evaluateType(unary.getRight());

        // If operator is pointer de-reference
        if (unary.getOperator().type == Token::Type::STAR) {
            auto rightPointerType = dynamic_cast<const Type::Pointer *>(rightType.type);
            if (!rightPointerType) {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType.type->getTypeName() + "'");
                throw TypeCheckError();
            }

            // Return value type
            m_QualifiedType = Type::QualifiedType{rightPointerType->getValueType(), rightType.constValue, false};
        }
        // Otherwise
        else {
            auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType.type);
            if (rightNumericType) {
                // If operator is arithmetic, return promoted type
                if (unary.getOperator().type == Token::Type::PLUS || unary.getOperator().type == Token::Type::MINUS) {
                    m_QualifiedType = Type::QualifiedType{Type::getPromotedType(rightNumericType),
                                                          rightType.constValue, false};
                }
                // Otherwise, if operator is bitwise
                else if (unary.getOperator().type == Token::Type::TILDA) {
                    // If type is integer, return promoted type
                    if (rightNumericType->isIntegral()) {
                        m_QualifiedType = Type::QualifiedType{Type::getPromotedType(rightNumericType),
                                                              rightType.constValue, false};
                    }
                    else {
                        m_ErrorHandler.error(unary.getOperator(),
                                             "Invalid operand type '" + rightType.type->getTypeName() + "'");
                        throw TypeCheckError();
                    }
                }
                // Otherwise, if operator is logical
                else if (unary.getOperator().type == Token::Type::NOT) {
                    m_QualifiedType = Type::QualifiedType{Type::Int32::getInstance(),
                                                          rightType.constValue, false};
                }
                // Otherwise, if operator is address of, return pointer type
                else if (unary.getOperator().type == Token::Type::AMPERSAND) {
                    m_QualifiedType = Type::QualifiedType{Type::createPointer(rightType.type),
                                                          rightType.constValue, false};
                }
            }
            else {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType.type->getTypeName() + "'");
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
            auto valNumericType = dynamic_cast<const Type::NumericBase *>(valType.type);
            if (!valNumericType || !valNumericType->isIntegral()) {
                m_ErrorHandler.error(labelled.getKeyword(),
                                     "Invalid case value '" + valType.type->getTypeName() + "'");
                throw TypeCheckError();
            }
        }

        labelled.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Switch &switchStatement) final
    {
        auto condType = evaluateType(switchStatement.getCondition());
        auto condNumericType = dynamic_cast<const Type::NumericBase *>(condType.type);
        if (!condNumericType || !condNumericType->isIntegral()) {
            m_ErrorHandler.error(switchStatement.getSwitch(),
                                 "Invalid condition '" + condType.type->getTypeName() + "'");
            throw TypeCheckError();
        }

        m_InSwitch = true;
        switchStatement.getBody()->accept(*this);
        m_InSwitch = false;
    }

    virtual void visit(const Statement::VarDeclaration &varDeclaration) final
    {
        for (const auto &var : varDeclaration.getInitDeclaratorList()) {
            m_Environment->define(std::get<0>(var), varDeclaration.getQualifiedType(), m_ErrorHandler);

            // If variable has an initialiser expression
            if (std::get<1>(var)) {
                // Evaluate type
                const auto initialiserType = evaluateType(std::get<1>(var).get());

                // Assign initialiser expression to variable
                m_Environment->assign(std::get<0>(var), Token::Type::EQUAL, initialiserType, m_ErrorHandler, true);
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
    const Type::QualifiedType &evaluateType(const Expression::Base *expression)
    {
        expression->accept(*this);
        return m_QualifiedType;
    }

    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentInternal *m_Environment;
    Type::QualifiedType m_QualifiedType;

    ErrorHandlerBase &m_ErrorHandler;
    bool m_InLoop;
    bool m_InSwitch;
};
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentBase
//---------------------------------------------------------------------------
const Type::QualifiedType &EnvironmentBase::assign(const Token &name, Token::Type op, 
                                                   const Type::QualifiedType &existingType, const Type::QualifiedType &assignedType, 
                                                   ErrorHandlerBase &errorHandler, bool initializer) const
{
    // If existing type is a constant numeric value or if it's a constant pointer give errors
    auto numericExistingType = dynamic_cast<const Type::NumericBase *>(existingType.type);
    auto pointerExistingType = dynamic_cast<const Type::Pointer *>(existingType.type);
    if(!initializer && ((numericExistingType && existingType.constValue) 
                        || (pointerExistingType && existingType.constPointer))) 
    {
        errorHandler.error(name, "Assignment of read-only variable");
        throw TypeCheckError();
    }
    
    // If assignment operation is plain equals, any type is fine so return
    auto numericAssignedType = dynamic_cast<const Type::NumericBase *>(assignedType.type);
    auto pointerAssignedType = dynamic_cast<const Type::Pointer *>(assignedType.type);
    if(op == Token::Type::EQUAL) {
        // If we're initialising a pointer with another pointer
        if (pointerAssignedType && pointerExistingType) {
            // If we're trying to assign a pointer to a const value to a pointer
            if (assignedType.constValue && !existingType.constValue) {
                errorHandler.error(name, "Invalid operand types '" + pointerExistingType->getTypeName() + "' and '" + pointerAssignedType->getTypeName());
                throw TypeCheckError();
            }

            // If pointer types aren't compatible
            if (pointerExistingType->getTypeName() != pointerAssignedType->getTypeName()) {
                errorHandler.error(name, "Invalid operand types '" + pointerExistingType->getTypeName() + "' and '" + pointerAssignedType->getTypeName());
                throw TypeCheckError();
            }
        }
        // Otherwise, if we're trying to initialise a pointer with a non-pointer or vice-versa
        else if (pointerAssignedType || pointerExistingType) {
            errorHandler.error(name, "Invalid operand types '" + existingType.type->getTypeName() + "' and '" + assignedType.type->getTypeName());
            throw TypeCheckError();
        }
    }
    // Otherwise, if operation is += or --
    else if (op == Token::Type::PLUS_EQUAL || op == Token::Type::MINUS_EQUAL) {
        // If the operand being added isn't numeric or the type being added to is neither numeric or a pointer
        if (!numericAssignedType || (!pointerExistingType && !numericExistingType))
        {
            errorHandler.error(name, "Invalid operand types '" + existingType.type->getTypeName() + "' and '" + assignedType.type->getTypeName() + "'");
            throw TypeCheckError();
        }

        // If we're adding a numeric type to a pointer, check it's an integer
        if (pointerExistingType && numericAssignedType->isIntegral()) {
            errorHandler.error(name, "Invalid operand types '" + numericAssignedType->getTypeName() + "'");
            throw TypeCheckError();
        }
    }
    // Otherwise, numeric types are required
    else {
        // If either type is non-numeric, give error
        if(!numericAssignedType) {
            errorHandler.error(name, "Invalid operand types '" + numericAssignedType->getTypeName() + "'");
            throw TypeCheckError();
        }
        if(!numericExistingType) {
            errorHandler.error(name, "Invalid operand types '" + existingType.type->getTypeName() + "'");
            throw TypeCheckError();
        }

        // If operand isn't one that takes any numeric type, check both operands are integral
        if (op != Token::Type::STAR_EQUAL && op != Token::Type::SLASH_EQUAL) {
            if(!numericAssignedType->isIntegral()) {
                errorHandler.error(name, "Invalid operand types '" + numericAssignedType->getTypeName() + "'");
                throw TypeCheckError();
            }
            if(!numericExistingType->isIntegral()) {
                errorHandler.error(name, "Invalid operand types '" + numericExistingType->getTypeName() + "'");
                throw TypeCheckError();
            }
        }
    }
   
     // Return existing type
     // **THINK**
    return existingType;
}
//---------------------------------------------------------------------------
const Type::QualifiedType &EnvironmentBase::incDec(const Token &name, Token::Type, 
                                                   const Type::QualifiedType &existingType, ErrorHandlerBase &errorHandler) const
{
    // If existing type is a constant numeric value or if it's a constant pointer give errors
    auto numericExistingType = dynamic_cast<const Type::NumericBase *>(existingType.type);
    auto pointerExistingType = dynamic_cast<const Type::Pointer *>(existingType.type);
    if((numericExistingType && existingType.constValue) 
        || (pointerExistingType && existingType.constPointer)) 
    {
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
                                              ErrorHandlerBase &errorHandler)
{
    Visitor visitor(errorHandler);
    EnvironmentInternal internalEnvironment(environment);
    visitor.typeCheck(statements, internalEnvironment);
}
//---------------------------------------------------------------------------
Type::QualifiedType GeNN::Transpiler::TypeChecker::typeCheck(const Expression::Base *expression, 
                                                             EnvironmentBase &environment,
                                                             ErrorHandlerBase &errorHandler)
{
    Visitor visitor(errorHandler);
    EnvironmentInternal internalEnvironment(environment);
    return visitor.typeCheck(expression, internalEnvironment);
}
