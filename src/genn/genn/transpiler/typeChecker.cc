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
// Vistor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(ErrorHandler &errorHandler)
    :   m_Environment(nullptr), m_QualifiedType{nullptr, false, false}, m_ErrorHandler(errorHandler), 
        m_InLoop(false), m_InSwitch(false)
    {
    }

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    // **THINK** make constructors?
    void typeCheck(const Statement::StatementList &statements, Environment &environment)
    {
        Environment *previous = m_Environment;
        m_Environment = &environment;
        for (auto &s : statements) {
            s.get()->accept(*this);
        }
        m_Environment = previous;
    }

    const Type::QualifiedType typeCheck(const Expression::Base *expression, Environment &environment)
    {
        Environment *previous = m_Environment;
        m_Environment = &environment;

        const auto type = evaluateType(expression);
        
        m_Environment = previous;
        return type;
    }

    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript &arraySubscript) final
    {
        // Get pointer type
        auto arrayType = m_Environment->getType(arraySubscript.getPointerName(), m_ErrorHandler);
        auto pointerType = dynamic_cast<const Type::NumericPtrBase *>(arrayType.type);

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
        m_QualifiedType = m_Environment->assign(assignment.getVarName(), rhsType,
                                                assignment.getOperator().type, m_ErrorHandler);
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
            auto leftNumericPtrType = dynamic_cast<const Type::NumericPtrBase *>(leftType.type);
            auto rightNumericPtrType = dynamic_cast<const Type::NumericPtrBase *>(rightType.type);
            if (leftNumericPtrType && rightNumericPtrType && opType == Token::Type::MINUS) {
                // Check pointers are compatible
                if (leftNumericPtrType->getTypeHash() != rightNumericPtrType->getTypeHash()) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType.type->getTypeName() + "' and '" + rightType.type->getTypeName());
                    throw TypeCheckError();
                }

                // **TODO** should be std::ptrdiff/Int64
                m_QualifiedType = Type::QualifiedType{Type::Int32::getInstance(), false, false};
            }
            // Otherwise, if we're adding to or subtracting from pointers
            else if (leftNumericPtrType && rightNumericType && (opType == Token::Type::PLUS || opType == Token::Type::MINUS))       // P + n or P - n
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
            else if (leftNumericType && rightNumericPtrType && opType == Token::Type::PLUS)  // n + P
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
        // **TODO** any numeric can be cast to any numeric and any pointer to pointer but no intermixing
        // **TODO** const cannot be removed like this
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
                                                postfixIncDec.getOperator(), m_ErrorHandler);
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        m_QualifiedType = m_Environment->incDec(prefixIncDec.getVarName(),
                                                prefixIncDec.getOperator(), m_ErrorHandler);
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
            auto rightNumericPtrType = dynamic_cast<const Type::NumericPtrBase *>(rightType.type);
            if (!rightNumericPtrType) {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType.type->getTypeName() + "'");
                throw TypeCheckError();
            }

            // Return value type
            m_QualifiedType = Type::QualifiedType{rightNumericPtrType->getValueType(), rightType.constValue, false};
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
                    m_QualifiedType = Type::QualifiedType{rightNumericType->getPointerType(),
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
        Environment environment(m_Environment);
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
        Environment *previous = m_Environment;
        Environment environment(m_Environment);
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
                // **TODO** flag to signify this is an initialiser
                m_Environment->assign(std::get<0>(var), initialiserType, Token::Type::EQUAL, m_ErrorHandler);
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
    Environment *m_Environment;
    Type::QualifiedType m_QualifiedType;

    ErrorHandler &m_ErrorHandler;
    bool m_InLoop;
    bool m_InSwitch;
};
}

//---------------------------------------------------------------------------
// MiniParse::TypeChecker::Environment
//---------------------------------------------------------------------------
void Environment::define(const Token &name, const Type::QualifiedType &qualifiedType, ErrorHandler &errorHandler)
{
    if(!m_Types.try_emplace(name.lexeme, qualifiedType).second) {
        errorHandler.error(name, "Redeclaration of variable");
        throw TypeCheckError();
    }
}
//---------------------------------------------------------------------------
const Type::QualifiedType &Environment::assign(const Token &name, const Type::QualifiedType &assignedType, 
                                               Token::Type op, ErrorHandler &errorHandler)
{
    // If type isn't found
    auto existingType = m_Types.find(name.lexeme);
    if(existingType == m_Types.end()) {
        if(m_Enclosing) {
            return m_Enclosing->assign(name, assignedType,
                                       op, errorHandler);
        }
        else {
            errorHandler.error(name, "Undefined variable");
            throw TypeCheckError();
        }
    }

    // If existing type is a constant numeric value or if it's a constant pointer give errors
    auto numericExistingType = dynamic_cast<const Type::NumericBase *>(existingType->second.type);
    auto numericPtrExistingType = dynamic_cast<const Type::NumericPtrBase *>(existingType->second.type);
    if((numericExistingType && existingType->second.constValue) 
        || (numericPtrExistingType && existingType->second.constPointer)) 
    {
        errorHandler.error(name, "Assignment of read-only variable");
        throw TypeCheckError();
    }
    
    // If assignment operation is plain equals, any type is fine so return
    auto numericAssignedType = dynamic_cast<const Type::NumericBase *>(assignedType.type);
    auto numericPtrAssignedType = dynamic_cast<const Type::NumericPtrBase *>(assignedType.type);
    // **TODO** pointer type check
    if(op == Token::Type::EQUAL) {
        // If we're initialising a pointer with another pointer
        if (numericPtrAssignedType && numericPtrExistingType) {
            // If variable is non-const but initialiser is const
            /*if (!varDeclaration.isConst() && intialiserConst) {
                m_ErrorHandler.error(std::get<0>(var),
                                        "Invalid operand types '" + initialiserType->getTypeName() + "'");
                throw TypeCheckError();
            }*/

            // If pointer types aren't compatible
            if (numericPtrExistingType->getTypeHash() != numericPtrAssignedType->getTypeHash()) {
                errorHandler.error(name, "Invalid operand types '" + numericPtrExistingType->getTypeName() + "' and '" + numericPtrAssignedType->getTypeName());
                throw TypeCheckError();
            }
        }
        // Otherwise, if we're trying to initialise a pointer with a non-pointer or vice-versa
        else if (numericPtrAssignedType || numericPtrExistingType) {
            errorHandler.error(name, "Invalid operand types '" + existingType->second.type->getTypeName() + "' and '" + assignedType.type->getTypeName());
            throw TypeCheckError();
        }
    }
    // Otherwise, if operation is += or --
    else if (op == Token::Type::PLUS_EQUAL || op == Token::Type::MINUS_EQUAL) {
        // If the operand being added isn't numeric or the type being added to is neither numeric or a pointer
        if (!numericAssignedType || (!numericPtrExistingType && !numericExistingType))
        {
            errorHandler.error(name, "Invalid operand types '" + existingType->second.type->getTypeName() + "' and '" + assignedType.type->getTypeName() + "'");
            throw TypeCheckError();
        }

        // If we're adding a numeric type to a pointer, check it's an integer
        if (numericPtrExistingType && numericAssignedType->isIntegral()) {
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
            errorHandler.error(name, "Invalid operand types '" + existingType->second.type->getTypeName() + "'");
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
    return existingType->second;
}
//---------------------------------------------------------------------------
const Type::QualifiedType &Environment::incDec(const Token &name, const Token &op, ErrorHandler &errorHandler)
{
    auto existingType = m_Types.find(name.lexeme);
    if(existingType == m_Types.end()) {
        if(m_Enclosing) {
            return m_Enclosing->incDec(name, op, errorHandler);
        }
        else {
            errorHandler.error(name, "Undefined variable");
            throw TypeCheckError();
        }
    }
    
    // If existing type is a constant numeric value or if it's a constant pointer give errors
    auto numericExistingType = dynamic_cast<const Type::NumericBase *>(existingType->second.type);
    auto numericPtrExistingType = dynamic_cast<const Type::NumericPtrBase *>(existingType->second.type);
    if((numericExistingType && existingType->second.constValue) 
        || (numericPtrExistingType && existingType->second.constPointer)) 
    {
        errorHandler.error(name, "Increment/decrement of read-only variable");
        throw TypeCheckError();
    }
    // Otherwise, return type
    else {
        return existingType->second;
    }
}
//---------------------------------------------------------------------------
const Type::QualifiedType &Environment::getType(const Token &name, ErrorHandler &errorHandler) const
{
    auto type = m_Types.find(std::string{name.lexeme});
    if(type == m_Types.end()) {
        if(m_Enclosing) {
            return m_Enclosing->getType(name, errorHandler);
        }
        else {
            errorHandler.error(name, "Undefined variable");
            throw TypeCheckError();
        }
    }
    else {
        return type->second;
    }
}
//---------------------------------------------------------------------------
void GeNN::Transpiler::TypeChecker::typeCheck(const Statement::StatementList &statements, Environment &environment, 
                                              ErrorHandler &errorHandler)
{
    Visitor visitor(errorHandler);
    visitor.typeCheck(statements, environment);
}
//---------------------------------------------------------------------------
Type::QualifiedType GeNN::Transpiler::TypeChecker::typeCheck(const Expression::Base *expression, 
                                                             Environment &environment,
                                                             ErrorHandler &errorHandler)
{
    Visitor visitor(errorHandler);
    return visitor.typeCheck(expression, environment);
}
