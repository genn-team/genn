#include "type_checker.h"

// Standard C++ includes
#include <string>

// Standard C includes
#include <cassert>

// GeNN includes
#include "type.h"

// Mini-parse includes
#include "error_handler.h"
#include "expression.h"
#include "utils.h"

using namespace MiniParse;
using namespace MiniParse::TypeChecker;

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
//---------------------------------------------------------------------------
// TypeCheckError
//---------------------------------------------------------------------------
class TypeCheckError
{
};

//---------------------------------------------------------------------------
// Vistor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(ErrorHandler &errorHandler)
        : m_Environment(nullptr), m_Type(nullptr), m_Const(false),
        m_ErrorHandler(errorHandler), m_InLoop(false), m_InSwitch(false)
    {
    }

    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    void typeCheck(const Statement::StatementList &statements, Environment &environment)
    {
        Environment *previous = m_Environment;
        m_Environment = &environment;
        for (auto &s : statements) {
            s.get()->accept(*this);
        }
        m_Environment = previous;
    }

    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript &arraySubscript) final
    {
        // Get pointer type
        auto pointerType = dynamic_cast<const Type::NumericPtrBase *>(
            std::get<0>(m_Environment->getType(arraySubscript.getPointerName(), m_ErrorHandler)));

        // If pointer is indeed a pointer
        if (pointerType) {
            // Evaluate pointer type
            auto indexType = evaluateType(arraySubscript.getIndex().get());
            auto indexNumericType = dynamic_cast<const Type::NumericBase *>(indexType);
            if (!indexNumericType || !indexNumericType->isIntegral()) {
                m_ErrorHandler.error(arraySubscript.getPointerName(),
                                     "Invalid subscript index type '" + indexType->getTypeName() + "'");
                throw TypeCheckError();
            }

            // Use value type of array
            m_Type = pointerType->getValueType();
            m_Const = false;
        }
        // Otherwise
        else {
            m_ErrorHandler.error(arraySubscript.getPointerName(), "Subscripted object is not a pointer");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Assignment &assignment) final
    {
        const auto [rhsType, rhsConst] = evaluateTypeConst(assignment.getValue());
        m_Type = m_Environment->assign(assignment.getVarName(), rhsType, rhsConst,
                                       assignment.getOperator().type, m_ErrorHandler);
        m_Const = false;
    }

    virtual void visit(const Expression::Binary &binary) final
    {
        const auto opType = binary.getOperator().type;
        const auto [rightType, rightConst] = evaluateTypeConst(binary.getRight());
        if (opType == Token::Type::COMMA) {
            m_Type = rightType;
            m_Const = rightConst;
        }
        else {
            // If we're subtracting two pointers
            const auto [leftType, leftConst] = evaluateTypeConst(binary.getLeft());
            auto leftNumericType = dynamic_cast<const Type::NumericBase *>(leftType);
            auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType);
            auto leftNumericPtrType = dynamic_cast<const Type::NumericPtrBase *>(leftType);
            auto rightNumericPtrType = dynamic_cast<const Type::NumericPtrBase *>(rightType);
            if (leftNumericPtrType && rightNumericPtrType && opType == Token::Type::MINUS) {
                // Check pointers are compatible
                if (leftNumericPtrType->getTypeHash() != rightNumericPtrType->getTypeHash()) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getTypeName() + "' and '" + rightType->getTypeName());
                    throw TypeCheckError();
                }

                // **TODO** should be std::ptrdiff/Int64
                m_Type = Type::Int32::getInstance();
                m_Const = false;
            }
            // Otherwise, if we're adding to or subtracting from pointers
            else if (leftNumericPtrType && rightNumericType && (opType == Token::Type::PLUS || opType == Token::Type::MINUS))       // P + n or P - n
            {
                // Check that numeric operand is integer
                if (!rightNumericType->isIntegral()) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getTypeName() + "' and '" + rightType->getTypeName());
                    throw TypeCheckError();
                }

                // Use pointer type
                m_Type = leftNumericPtrType;
                m_Const = leftConst;
            }
            // Otherwise, if we're adding a number to a pointer
            else if (leftNumericType && rightNumericPtrType && opType == Token::Type::PLUS)  // n + P
            {
                // Check that numeric operand is integer
                if (!leftNumericType->isIntegral()) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getTypeName() + "' and '" + rightType->getTypeName());
                    throw TypeCheckError();
                }

                // Use pointer type
                m_Type = rightNumericPtrType;
                m_Const = rightConst;
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
                        m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getTypeName() + "' and '" + rightType->getTypeName());
                        throw TypeCheckError();
                    }

                    // If operator is a shift, promote left type
                    if (opType == Token::Type::SHIFT_LEFT || opType == Token::Type::SHIFT_RIGHT) {
                        m_Type = Type::getPromotedType(leftNumericType);
                        m_Const = false;
                    }
                    // Otherwise, take common type
                    else {
                        m_Type = Type::getCommonType(leftNumericType, rightNumericType);
                        m_Const = false;
                    }
                }
                // Otherwise, any numeric type will do, take common type
                else {
                    m_Type = Type::getCommonType(leftNumericType, rightNumericType);
                    m_Const = false;
                }
            }
            else {
                m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getTypeName() + "' and '" + rightType->getTypeName());
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
                m_Const = false;
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
        m_Type = cast.getType();
        m_Const = cast.isConst();
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        const auto [trueType, trueConst] = evaluateTypeConst(conditional.getTrue());
        const auto [falseType, falseConst] = evaluateTypeConst(conditional.getFalse());
        auto trueNumericType = dynamic_cast<const Type::NumericBase *>(trueType);
        auto falseNumericType = dynamic_cast<const Type::NumericBase *>(falseType);
        if (trueNumericType && falseNumericType) {
            m_Type = Type::getCommonType(trueNumericType, falseNumericType);
            m_Const = trueConst || falseConst;
        }
        else {
            m_ErrorHandler.error(conditional.getQuestion(),
                                 "Invalid operand types '" + trueType->getTypeName() + "' and '" + std::string{falseType->getTypeName()} + "' to conditional");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Grouping &grouping) final
    {
        std::tie(m_Type, m_Const) = evaluateTypeConst(grouping.getExpression());
    }

    virtual void visit(const Expression::Literal &literal) final
    {
        m_Type = std::visit(
            MiniParse::Utils::Overload{
                [](auto v)->const Type::NumericBase *{ return Type::TypeTraits<decltype(v)>::NumericType::getInstance(); },
                [](std::monostate)->const Type::NumericBase *{ return nullptr; }},
                literal.getValue());
        m_Const = false;
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        logical.getLeft()->accept(*this);
        logical.getRight()->accept(*this);
        m_Type = Type::Int32::getInstance();
        m_Const = false;
    }

    virtual void visit(const Expression::PostfixIncDec &postfixIncDec) final
    {
        m_Type = m_Environment->incDec(postfixIncDec.getVarName(),
                                       postfixIncDec.getOperator(), m_ErrorHandler);
        m_Const = false;
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        m_Type = m_Environment->incDec(prefixIncDec.getVarName(),
                                       prefixIncDec.getOperator(), m_ErrorHandler);
        m_Const = false;
    }

    virtual void visit(const Expression::Variable &variable)
    {
        std::tie(m_Type, m_Const) = m_Environment->getType(variable.getName(), m_ErrorHandler);
    }

    virtual void visit(const Expression::Unary &unary) final
    {
        const auto [rightType, rightConst] = evaluateTypeConst(unary.getRight());

        // If operator is pointer de-reference
        if (unary.getOperator().type == Token::Type::STAR) {
            auto rightNumericPtrType = dynamic_cast<const Type::NumericPtrBase *>(rightType);
            if (!rightNumericPtrType) {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType->getTypeName() + "'");
                throw TypeCheckError();
            }

            // Return value type
            m_Type = rightNumericPtrType->getValueType();

            // **THINK**
            m_Const = false;
        }
        // Otherwise
        else {
            auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType);
            if (rightNumericType) {
                // If operator is arithmetic, return promoted type
                if (unary.getOperator().type == Token::Type::PLUS || unary.getOperator().type == Token::Type::MINUS) {
                    m_Type = Type::getPromotedType(rightNumericType);
                    m_Const = false;
                }
                // Otherwise, if operator is bitwise
                else if (unary.getOperator().type == Token::Type::TILDA) {
                    // If type is integer, return promoted type
                    if (rightNumericType->isIntegral()) {
                        m_Type = Type::getPromotedType(rightNumericType);
                        m_Const = false;
                    }
                    else {
                        m_ErrorHandler.error(unary.getOperator(),
                                             "Invalid operand type '" + rightType->getTypeName() + "'");
                        throw TypeCheckError();
                    }
                }
                // Otherwise, if operator is logical
                else if (unary.getOperator().type == Token::Type::NOT) {
                    m_Type = Type::Int32::getInstance();
                    m_Const = false;
                }
                // Otherwise, if operator is address of, return pointer type
                else if (unary.getOperator().type == Token::Type::AMPERSAND) {
                    m_Type = rightNumericType->getPointerType();
                    m_Const = rightConst;
                }
            }
            else {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType->getTypeName() + "'");
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
            auto valNumericType = dynamic_cast<const Type::NumericBase *>(valType);
            if (!valNumericType || !valNumericType->isIntegral()) {
                m_ErrorHandler.error(labelled.getKeyword(),
                                     "Invalid case value '" + valType->getTypeName() + "'");
                throw TypeCheckError();
            }
        }

        labelled.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Switch &switchStatement) final
    {
        auto condType = evaluateType(switchStatement.getCondition());
        auto condNumericType = dynamic_cast<const Type::NumericBase *>(condType);
        if (!condNumericType || !condNumericType->isIntegral()) {
            m_ErrorHandler.error(switchStatement.getSwitch(),
                                 "Invalid condition '" + condType->getTypeName() + "'");
            throw TypeCheckError();
        }

        m_InSwitch = true;
        switchStatement.getBody()->accept(*this);
        m_InSwitch = false;
    }

    virtual void visit(const Statement::VarDeclaration &varDeclaration) final
    {
        for (const auto &var : varDeclaration.getInitDeclaratorList()) {
            m_Environment->define(std::get<0>(var), varDeclaration.getType(),
                                  varDeclaration.isConst(), m_ErrorHandler);

            // If variable has an initialiser expression
            if (std::get<1>(var)) {
                // Evaluate type
                const auto [initialiserType, initialiserConst] = evaluateTypeConst(std::get<1>(var).get());

                // Assign initialiser expression to variable
                m_Environment->assign(std::get<0>(var), initialiserType, initialiserConst, Token::Type::EQUAL, m_ErrorHandler);
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
    std::tuple<const Type::Base *, bool> evaluateTypeConst(const Expression::Base *expression)
    {
        expression->accept(*this);
        return std::make_tuple(m_Type, m_Const);
    }

    const Type::Base *evaluateType(const Expression::Base *expression)
    {
        return std::get<0>(evaluateTypeConst(expression));
    }

    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    Environment *m_Environment;
    const Type::Base *m_Type;
    bool m_Const;

    ErrorHandler &m_ErrorHandler;
    bool m_InLoop;
    bool m_InSwitch;
};
}

//---------------------------------------------------------------------------
// MiniParse::TypeChecker::Environment
//---------------------------------------------------------------------------
void Environment::define(const Token &name, const Type::Base *type, bool isConst, ErrorHandler &errorHandler)
{
    if(!m_Types.try_emplace(name.lexeme, type, isConst).second) {
        errorHandler.error(name, "Redeclaration of variable");
        throw TypeCheckError();
    }
}
//---------------------------------------------------------------------------
const Type::Base *Environment::assign(const Token &name, const Type::Base *assignedType, bool assignedConst, 
                                      Token::Type op, ErrorHandler &errorHandler)
{
    // If type isn't found
    auto existingType = m_Types.find(name.lexeme);
    if(existingType == m_Types.end()) {
        if(m_Enclosing) {
            return m_Enclosing->assign(name, assignedType, 
                                       assignedConst, op, errorHandler);
        }
        else {
            errorHandler.error(name, "Undefined variable");
            throw TypeCheckError();
        }
    }
    // Otherwise, if type is found and it's const, give error
    else if(std::get<1>(existingType->second)) {
        errorHandler.error(name, "Assignment of read-only variable");
        throw TypeCheckError();
    }

    auto numericExistingType = dynamic_cast<const Type::NumericBase *>(std::get<0>(existingType->second));
    auto numericAssignedType = dynamic_cast<const Type::NumericBase *>(assignedType);

    auto numericPtrExistingType = dynamic_cast<const Type::NumericPtrBase *>(std::get<0>(existingType->second));
    auto numericPtrAssignedType = dynamic_cast<const Type::NumericPtrBase *>(assignedType);

    // If assignment operation is plain equals, any type is fine so return
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
            errorHandler.error(name, "Invalid operand types '" + std::get<0>(existingType->second)->getTypeName() + "' and '" + assignedType->getTypeName());
            throw TypeCheckError();
        }
    }
    // Otherwise, if operation is += or --
    else if (op == Token::Type::PLUS_EQUAL || op == Token::Type::MINUS_EQUAL) {
        // If the operand being added isn't numeric or the type being added to is neither numeric or a pointer
        if (!numericAssignedType || (!numericPtrExistingType && !numericExistingType))
        {
            errorHandler.error(name, "Invalid operand types '" + std::get<0>(existingType->second)->getTypeName() + "' and '" + assignedType->getTypeName() + "'");
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
            errorHandler.error(name, "Invalid operand types '" + std::get<0>(existingType->second)->getTypeName() + "'");
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
    return std::get<0>(existingType->second);
}
//---------------------------------------------------------------------------
const Type::Base *Environment::incDec(const Token &name, const Token &op, ErrorHandler &errorHandler)
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
    // Otherwise, if type is found and it's const, give error
    else if(std::get<1>(existingType->second)) {
        errorHandler.error(name, "Increment/decrement of read-only variable");
        throw TypeCheckError();
    }
    // Otherwise, return type
    // **TODO** pointer
    else {
        auto numericExistingType = dynamic_cast<const Type::NumericBase *>(std::get<0>(existingType->second));
        if(numericExistingType == nullptr) {
            errorHandler.error(op, "Invalid operand types '" + std::get<0>(existingType->second)->getTypeName() + "'");
            throw TypeCheckError();
        }
        else {
            return std::get<0>(existingType->second);
        }
    }
}
//---------------------------------------------------------------------------
std::tuple<const Type::Base *, bool> Environment::getType(const Token &name, ErrorHandler &errorHandler) const
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
void MiniParse::TypeChecker::typeCheck(const Statement::StatementList &statements, Environment &environment, 
                                       ErrorHandler &errorHandler)
{
    Visitor visitor(errorHandler);
    visitor.typeCheck(statements, environment);
}