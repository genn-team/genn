#include "transpiler/typeChecker.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>
#include <string>
#include <iostream>
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
bool checkPointerTypeAssignement(const Type::Base *rightType, const Type::Base *leftType, const Type::TypeContext &typeContext) 
{
    // If both are pointers, recurse through value type
    auto rightPointerType = dynamic_cast<const Type::Pointer *>(rightType);
    auto leftPointerType = dynamic_cast<const Type::Pointer *>(leftType);
    if (rightPointerType && leftPointerType) {
        return checkPointerTypeAssignement(rightPointerType->getValueType(), leftPointerType->getValueType(), typeContext);
    }
    // Otherwise, if we've hit the value type at the end of the chain, check resolved names match
    else if (!rightPointerType && !leftPointerType) {
        return (rightType->getResolvedName(typeContext) == leftType->getResolvedName(typeContext));
    }
    // Otherwise, pointers with different levels of indirection e.g. int* and int** are being compared
    else {
        return false;
    }
}
//---------------------------------------------------------------------------
bool checkForConstRemoval(const Type::Base *rightType, const Type::Base *leftType) 
{
    // If const is being removed
    if (rightType->hasQualifier(Type::Qualifier::CONSTANT) && !leftType->hasQualifier(Type::Qualifier::CONSTANT)) {
        return false;
    }

    // If both are pointers, recurse through value type
    auto rightPointerType = dynamic_cast<const Type::Pointer *>(rightType);
    auto leftPointerType = dynamic_cast<const Type::Pointer *>(leftType);
    if (rightPointerType && leftPointerType) {
        return checkForConstRemoval(rightPointerType->getValueType(), leftPointerType->getValueType());
    }
    // Otherwise, if both are non-pointers, return true as const removal has been succesfully checked
    else if (!rightPointerType && !leftPointerType) {
        return true;
    }
    // Otherwise, pointers with different levels of indirection e.g. int* and int** are being compared
    else {
        return false;
    }

}

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

    virtual std::vector<const Type::Base*> getTypes(const Token &name, ErrorHandlerBase &errorHandler) final
    {
        auto type = m_Types.find(name.lexeme);
        if(type == m_Types.end()) {
            return m_Enclosing.getTypes(name, errorHandler);
        }
        else {
            return {type->second};
        }
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    EnvironmentBase &m_Enclosing;
    std::unordered_map<std::string, const Type::Base*> m_Types;
};

//---------------------------------------------------------------------------
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor, public Statement::Visitor
{
public:
    Visitor(const Statement::StatementList &statements, const Type::TypeContext &context, 
            EnvironmentInternal &environment, ResolvedTypeMap &resolvedTypes, ErrorHandlerBase &errorHandler)
    :   Visitor(context, environment, resolvedTypes, errorHandler)
    {
        for (auto &s : statements) {
            s.get()->accept(*this);
        }
    }
    
    Visitor(const Expression::Base *expression, const Type::TypeContext &context, 
            EnvironmentInternal &environment, ResolvedTypeMap &resolvedTypes, ErrorHandlerBase &errorHandler)
    :   Visitor(context, environment, resolvedTypes, errorHandler)
    {
        expression->accept(*this);
    }
    
private:
    Visitor(const Type::TypeContext &context, EnvironmentInternal &environment, 
            ResolvedTypeMap &resolvedTypes, ErrorHandlerBase &errorHandler)
    :   m_Environment(environment), m_Context(context), m_ErrorHandler(errorHandler), 
        m_ResolvedTypes(resolvedTypes), m_InLoop(false), m_InSwitch(false)
    {
    }

    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript &arraySubscript) final
    {
        // Get pointer type
        auto arrayType = m_Environment.get().getType(arraySubscript.getPointerName(), m_ErrorHandler);
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
            setExpressionType(&arraySubscript, pointerType->getValueType());
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
        setExpressionType(&assignment,
                          m_Environment.get().assign(assignment.getVarName(), assignment.getOperator().type, rhsType, 
                                                     m_Context, m_ErrorHandler));
    }

    virtual void visit(const Expression::Binary &binary) final
    {
        const auto opType = binary.getOperator().type;
        const auto rightType = evaluateType(binary.getRight());
        if (opType == Token::Type::COMMA) {
            setExpressionType(&binary, rightType);
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
                setExpressionType<Type::Int32>(&binary);
            }
            // Otherwise, if we're adding to or subtracting from pointers
            else if (leftPointerType && rightNumericType && (opType == Token::Type::PLUS || opType == Token::Type::MINUS)) {    // P + n or P - n
                // Check that numeric operand is integer
                if (!rightNumericType->isIntegral(m_Context)) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getName() + "' and '" + rightType->getName());
                    throw TypeCheckError();
                }

                // Use left type
                setExpressionType(&binary, leftType);
            }
            // Otherwise, if we're adding a number to a pointer
            else if (leftNumericType && rightPointerType && opType == Token::Type::PLUS) {  // n + P
                // Check that numeric operand is integer
                if (!leftNumericType->isIntegral(m_Context)) {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType->getName() + "' and '" + rightType->getName());
                    throw TypeCheckError();
                }

                // Use right type
                setExpressionType(&binary, rightType);
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
                        setExpressionType(&binary, Type::getPromotedType(leftNumericType, m_Context));
                    }
                    // Otherwise, take common type
                    else {
                        setExpressionType(&binary, Type::getCommonType(leftNumericType, rightNumericType, m_Context));
                    }
                }
                // Otherwise, any numeric type will do, take common type
                else {
                    setExpressionType(&binary, Type::getCommonType(leftNumericType, rightNumericType, m_Context));
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
        // **TODO** think about nested calls
        assert(m_CallArguments.empty());

        // Evaluate argument types and store in class
        m_CallArguments.clear();
        std::transform(call.getArguments().cbegin(), call.getArguments().cend(), std::back_inserter(m_CallArguments),
                       [this](const auto &a){ return evaluateType(a.get()); });

        // Evaluate callee type
        auto calleeType = evaluateType(call.getCallee());
        auto calleeFunctionType = dynamic_cast<const Type::FunctionBase *>(calleeType);

        m_CallArguments.clear();
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
                setExpressionType(&call, calleeFunctionType->getReturnType());
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
        if (!checkForConstRemoval(rightType, cast.getType())) {
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getType()->getName() + "' and '" + rightType->getName());
            throw TypeCheckError();
        }

        // If we're trying to cast pointer to pointer
        auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType);
        auto rightPointerType = dynamic_cast<const Type::Pointer *>(rightType);
        auto leftNumericType = dynamic_cast<const Type::NumericBase *>(cast.getType());
        auto leftPointerType = dynamic_cast<const Type::Pointer *>(cast.getType());
        if (rightPointerType && leftPointerType) {
            // Check that value type at the end matches
            if (!checkPointerTypeAssignement(rightPointerType->getValueType(), leftPointerType->getValueType(), m_Context)) {
                m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getType()->getName() + "' and '" + rightType->getName());
                throw TypeCheckError();
            }
        }
        // Otherwise, if either operand isn't numeric
        else if(!leftNumericType | !rightNumericType) {
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getType()->getName() + "' and '" + rightType->getName());
            throw TypeCheckError();
        }

        setExpressionType(&cast, cast.getType());
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        const auto trueType = evaluateType(conditional.getTrue());
        const auto falseType = evaluateType(conditional.getFalse());
        auto trueNumericType = dynamic_cast<const Type::NumericBase *>(trueType);
        auto falseNumericType = dynamic_cast<const Type::NumericBase *>(falseType);
        if (trueNumericType && falseNumericType) {
            // **TODO** check behaviour
            const Type::Base *type = Type::getCommonType(trueNumericType, falseNumericType, m_Context);
            if(trueType->hasQualifier(Type::Qualifier::CONSTANT) || falseType->hasQualifier(Type::Qualifier::CONSTANT)) {
                type = type->getQualifiedType(Type::Qualifier::CONSTANT);
            }
            setExpressionType(&conditional, type);
        }
        else {
            m_ErrorHandler.error(conditional.getQuestion(),
                                 "Invalid operand types '" + trueType->getName() + "' and '" + falseType->getName() + "' to conditional");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Grouping &grouping) final
    {
        setExpressionType(&grouping, evaluateType(grouping.getExpression()));
    }

    virtual void visit(const Expression::Literal &literal) final
    {
        // Convert number token type to type
        // **THINK** is it better to use typedef for scalar or resolve from m_Context
        if (literal.getValue().type == Token::Type::DOUBLE_NUMBER) {
            setExpressionType<Type::Double>(&literal);
        }
        else if (literal.getValue().type == Token::Type::FLOAT_NUMBER) {
            setExpressionType<Type::Float>(&literal);
        }
        else if (literal.getValue().type == Token::Type::SCALAR_NUMBER) {
            // **TODO** cache
            setExpressionType(&literal, new Type::NumericTypedef("scalar"));
        }
        else if (literal.getValue().type == Token::Type::INT32_NUMBER) {
            setExpressionType<Type::Int32>(&literal);
        }
        else if (literal.getValue().type == Token::Type::UINT32_NUMBER) {
            setExpressionType<Type::Uint32>(&literal);
        }
        else {
            assert(false);
        }
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        logical.getLeft()->accept(*this);
        logical.getRight()->accept(*this);
        setExpressionType<Type::Int32>(&logical);
    }

    virtual void visit(const Expression::PostfixIncDec &postfixIncDec) final
    {
        setExpressionType(&postfixIncDec, 
                          m_Environment.get().incDec(postfixIncDec.getVarName(), postfixIncDec.getOperator().type, 
                                                     m_Context, m_ErrorHandler));
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        setExpressionType(&prefixIncDec,
                          m_Environment.get().incDec(prefixIncDec.getVarName(), prefixIncDec.getOperator().type, 
                                                     m_Context, m_ErrorHandler));
    }

    virtual void visit(const Expression::Variable &variable)
    {
        // If type of variable is unambiguous, 
        const auto varTypes = m_Environment.get().getTypes(variable.getName(), m_ErrorHandler);
        if (varTypes.size() == 1) {
            setExpressionType(&variable, varTypes.front());
        }
        // Otherwise
        else {
            // Loop through variable types
            std::vector<std::pair<const Type::FunctionBase*, std::vector<int>>> viableFunctions;
            for(const auto *type : varTypes) {
                // Cast to function (only functions should be overloaded)
                const auto *func = dynamic_cast<const Type::FunctionBase*>(type);
                assert(func);

                // If number of arguments match
                const auto argumentTypes = func->getArgumentTypes();
                if(argumentTypes.size() == m_CallArguments.size()) {
                    // Create vector to hold argument conversion rank
                    std::vector<int> argumentConversionRank;
                    argumentConversionRank.reserve(m_CallArguments.size());

                    // Loop through arguments
                    bool viable = true;
                    auto c = m_CallArguments.cbegin();
                    auto a = argumentTypes.cbegin();
                    for(;c != m_CallArguments.cend(); c++, a++) {
                        auto cNumericType = dynamic_cast<const Type::NumericBase *>(*c);
                        auto aNumericType = dynamic_cast<const Type::NumericBase *>(*a);

                        // If both are numeric
                        if(cNumericType && aNumericType) {
                            // If names are identical (we don't care about qualifiers), match is exact
                            if(cNumericType->getName() == aNumericType->getName()) {
                                argumentConversionRank.push_back(0);
                            }
                            // Integer promotion
                            else if(aNumericType->getName() == Type::Int32::getInstance()->getName()
                                    && cNumericType->isIntegral(m_Context)
                                    && cNumericType->getRank(m_Context) < Type::Int32::getInstance()->getRank(m_Context))
                            {
                                argumentConversionRank.push_back(1);
                            }
                            // Float promotion
                            else if(aNumericType->getName() == Type::Double::getInstance()->getName()
                                        && cNumericType->getName() == Type::Float::getInstance()->getName())
                            {
                                argumentConversionRank.push_back(1);
                            }
                            // Otherwise, numeric conversion
                            else {
                                argumentConversionRank.push_back(2);
                            }
                        }
                        // Otherwise, if they are matching pointers
                        // **TODO** some more nuance here
                        else if(checkPointerTypeAssignement(*c, *a, m_Context)) {
                            argumentConversionRank.push_back(0);
                        }
                        // Otherwise, this function is not viable
                        else {
                            viable = false;
                            break;
                        }
                    }

                    // If function is viable, add to vector along with vector of conversion ranks
                    if(viable) {
                        assert(argumentConversionRank.size() == m_CallArguments.size());
                        viableFunctions.emplace_back(func, argumentConversionRank);
                    }
                }
            }

            if(viableFunctions.empty()) {
                m_ErrorHandler.error(variable.getName(),
                                        "No viable function candidates for '" + variable.getName().lexeme + "'");
                throw TypeCheckError();
            }
            else {
                std::cout << viableFunctions.size() << " function candidates" << std::endl;;
            }


            // **TODO** handler overload resolution
            assert(false);
        }
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
            setExpressionType(&unary, rightPointerType->getValueType());
        }
        // Otherwise
        else {
            auto rightNumericType = dynamic_cast<const Type::NumericBase *>(rightType);
            if (rightNumericType) {
                // If operator is arithmetic, return promoted type
                if (unary.getOperator().type == Token::Type::PLUS || unary.getOperator().type == Token::Type::MINUS) {
                    // **THINK** const through these?
                    setExpressionType(&unary, Type::getPromotedType(rightNumericType, m_Context));
                }
                // Otherwise, if operator is bitwise
                else if (unary.getOperator().type == Token::Type::TILDA) {
                    // If type is integer, return promoted type
                    if (rightNumericType->isIntegral(m_Context)) {
                        // **THINK** const through these?
                        setExpressionType(&unary, Type::getPromotedType(rightNumericType, m_Context));
                    }
                    else {
                        m_ErrorHandler.error(unary.getOperator(),
                                             "Invalid operand type '" + rightType->getName() + "'");
                        throw TypeCheckError();
                    }
                }
                // Otherwise, if operator is logical
                else if (unary.getOperator().type == Token::Type::NOT) {
                    setExpressionType<Type::Int32>(&unary);
                }
                // Otherwise, if operator is address of, return pointer type
                else if (unary.getOperator().type == Token::Type::AMPERSAND) {
                    setExpressionType(&unary, rightType->getPointerType());
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
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentInternal> oldEnvironment = m_Environment; 
        
        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;
        
        for (auto &s : compound.getStatements()) {
            s.get()->accept(*this);
        }
        
        // Restore old environment
        m_Environment = oldEnvironment;
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
        // Cache reference to current reference
        std::reference_wrapper<EnvironmentInternal> oldEnvironment = m_Environment; 
        
        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;

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

        // Restore old environment
        m_Environment = oldEnvironment;
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
            m_Environment.get().define(std::get<0>(var), varDeclaration.getType(), m_ErrorHandler);

            // If variable has an initialiser expression
            if (std::get<1>(var)) {
                // Evaluate type
                const auto initialiserType = evaluateType(std::get<1>(var).get());

                // Assign initialiser expression to variable
                m_Environment.get().assign(std::get<0>(var), Token::Type::EQUAL, initialiserType, 
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
        return m_ResolvedTypes.at(expression);
    }
   
    void setExpressionType(const Expression::Base *expression, const Type::Base *type)
    {
        if (!m_ResolvedTypes.emplace(expression, type).second) {
            throw std::runtime_error("Expression type resolved multiple times");
        }
    }

    template<typename T>
    void setExpressionType(const Expression::Base *expression)
    {
        if (!m_ResolvedTypes.emplace(expression, T::getInstance()).second) {
            throw std::runtime_error("Expression type resolved multiple times");
        }
    }
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::reference_wrapper<EnvironmentInternal> m_Environment;
    const Type::TypeContext &m_Context;
    ErrorHandlerBase &m_ErrorHandler;
    ResolvedTypeMap &m_ResolvedTypes;
    std::vector<const Type::Base*> m_CallArguments;
    bool m_InLoop;
    bool m_InSwitch;
};
}   // Anonymous namespace

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentBase
//---------------------------------------------------------------------------
const Type::Base *EnvironmentBase::getType(const Token &name, ErrorHandlerBase &errorHandler)
{
    const auto types = getTypes(name, errorHandler);
    if (types.size() == 1) {
        return types.front();
    }
    else {
        errorHandler.error(name, "Unambiguous type expected");
        throw TypeCheckError();
    }
}
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
            // Check that value type at the end matches
            if (!checkPointerTypeAssignement(pointerAssignedType->getValueType(), pointerExistingType->getValueType(), context)) {
                errorHandler.error(name, "Invalid operand types '" + pointerExistingType->getName() + "' and '" + pointerAssignedType->getName());
                throw TypeCheckError();
            }

            // If we're trying to make type less const
            if (!checkForConstRemoval(pointerAssignedType, pointerExistingType)) {
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
// GeNN::Transpiler::TypeChecker::StandardLibraryFunctionEnvironment
//---------------------------------------------------------------------------
#define ADD_FLOAT_DOUBLE(NAME, CLASS_PREFIX) {#NAME, Type::CLASS_PREFIX##F::getInstance()}, {#NAME, Type::CLASS_PREFIX##D::getInstance()}
StandardLibraryFunctionEnvironment::StandardLibraryFunctionEnvironment()
    : m_Types{ADD_FLOAT_DOUBLE(cos, Cos), ADD_FLOAT_DOUBLE(sin, Sin), ADD_FLOAT_DOUBLE(tan, Tan),
              ADD_FLOAT_DOUBLE(acos, Acos), ADD_FLOAT_DOUBLE(asin, Asin), ADD_FLOAT_DOUBLE(atan, Atan), ADD_FLOAT_DOUBLE(atan2, Atan2),
              ADD_FLOAT_DOUBLE(cosh, Cosh), ADD_FLOAT_DOUBLE(sinh, Sinh), ADD_FLOAT_DOUBLE(tanh, Tanh),
              ADD_FLOAT_DOUBLE(exp, Exp), ADD_FLOAT_DOUBLE(expm1, ExpM1), ADD_FLOAT_DOUBLE(exp2, Exp2), ADD_FLOAT_DOUBLE(pow, Pow),
              ADD_FLOAT_DOUBLE(scalbn, ScalBN), ADD_FLOAT_DOUBLE(log, Log), ADD_FLOAT_DOUBLE(log1p, Log1P), ADD_FLOAT_DOUBLE(log2, Log2), 
              ADD_FLOAT_DOUBLE(log10, Log10), ADD_FLOAT_DOUBLE(ldexp, LdExp), ADD_FLOAT_DOUBLE(ilogb, ILogB),
              ADD_FLOAT_DOUBLE(sqrt, Sqrt), ADD_FLOAT_DOUBLE(cbrt, Cbrt), ADD_FLOAT_DOUBLE(hypot, Hypot),
              ADD_FLOAT_DOUBLE(ceil, Ceil), ADD_FLOAT_DOUBLE(floor, Floor), ADD_FLOAT_DOUBLE(fmod, Fmod),
              ADD_FLOAT_DOUBLE(round, Round), ADD_FLOAT_DOUBLE(rint, Rint), ADD_FLOAT_DOUBLE(trunc, Trunc),
              ADD_FLOAT_DOUBLE(nearbyint, NearbyInt), ADD_FLOAT_DOUBLE(nextafter, NextAfter),ADD_FLOAT_DOUBLE(remainder, Remainder),
              ADD_FLOAT_DOUBLE(fabs, FAbs), ADD_FLOAT_DOUBLE(fdim, FDim), ADD_FLOAT_DOUBLE(fmax, FMax), ADD_FLOAT_DOUBLE(fmin, FMin),
              ADD_FLOAT_DOUBLE(erf, Erf), ADD_FLOAT_DOUBLE(erfc, ErfC), ADD_FLOAT_DOUBLE(tgamma, TGamma), ADD_FLOAT_DOUBLE(lgamma, LGamma),
              ADD_FLOAT_DOUBLE(copysign, CopySign), ADD_FLOAT_DOUBLE(fma, FMA)}
{
}
#undef ADD_FLOAT_DOUBLE
//------------------------------------------------------------------------
void StandardLibraryFunctionEnvironment::define(const Token &name, const Type::Base*, ErrorHandlerBase &errorHandler)
{
    errorHandler.error(name, "Cannot declare variable in external environment");
    throw TypeCheckError();
}
//---------------------------------------------------------------------------
const Type::Base *StandardLibraryFunctionEnvironment::assign(const Token &name, Token::Type, const Type::Base*,
                                                             const Type::TypeContext&, ErrorHandlerBase &errorHandler, bool)
{
    errorHandler.error(name, "Cannot assign variable in external environment");
    throw TypeCheckError();
}
//---------------------------------------------------------------------------
const Type::Base *StandardLibraryFunctionEnvironment::incDec(const Token &name, Token::Type, const Type::TypeContext&, 
                                                             ErrorHandlerBase &errorHandler)
{
    errorHandler.error(name, "Cannot increment/decrement variable in external environment");
    throw TypeCheckError();
}
//---------------------------------------------------------------------------
std::vector<const Type::Base*> StandardLibraryFunctionEnvironment::getTypes(const Token &name, ErrorHandlerBase &errorHandler)
{
    auto [typeBegin, typeEnd] = m_Types.equal_range(name.lexeme);
    if (typeBegin == typeEnd) {
         errorHandler.error(name, "Undefined variable");
         throw TypeCheckError();
    }
    else {
        std::vector<const Type::Base*> types;
        types.reserve(std::distance(typeBegin, typeEnd));
        std::transform(typeBegin, typeEnd, std::back_inserter(types),
                       [](auto t) { return t.second; });
        return types;
    }
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker
//---------------------------------------------------------------------------
void GeNN::Transpiler::TypeChecker::typeCheck(const Statement::StatementList &statements, EnvironmentBase &environment, 
                                              const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    ResolvedTypeMap expressionTypes;
    EnvironmentInternal internalEnvironment(environment);
    Visitor(statements, context, internalEnvironment, expressionTypes, errorHandler);
}
//---------------------------------------------------------------------------
const Type::Base *GeNN::Transpiler::TypeChecker::typeCheck(const Expression::Base *expression, EnvironmentBase &environment,
                                                           const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    ResolvedTypeMap expressionTypes;
    EnvironmentInternal internalEnvironment(environment);
    Visitor visitor(expression, context, internalEnvironment, expressionTypes, errorHandler);
    return expressionTypes.at(expression);
}
