#include "transpiler/typeChecker.h"

// Standard C++ includes
#include <algorithm>
#include <iterator>
#include <optional>
#include <stack>
#include <string>

// Standard C includes
#include <cassert>

// GeNN includes
#include "gennUtils.h"
#include "type.h"

// Transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/expression.h"

using namespace GeNN::Transpiler;
using namespace GeNN::Transpiler::TypeChecker;
namespace Type = GeNN::Type;
namespace Utils = GeNN::Utils;

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
std::string getDescription(const Type::ResolvedType &type)
{
    const std::string qualifier = type.hasQualifier(Type::Qualifier::CONSTANT) ? "const " : "";
     return std::visit(
         Utils::Overload{
             [&qualifier](const Type::ResolvedType::Value &value)
             {
                 assert(value.numeric);
                 return qualifier + value.name;
             },
             [&qualifier, &type](const Type::ResolvedType::Pointer &pointer)
             {
                 return qualifier + getDescription(*pointer.valueType) + "*";
             },
             [&type](const Type::ResolvedType::Function &function)
             {
                 std::string description = getDescription(*function.returnType) + "(";
                 for (const auto &a : function.argTypes) {
                     description += (getDescription(a) + ",");
                 }
                 return description + ")";
             }},
        type.detail);
}
//---------------------------------------------------------------------------
bool checkPointerTypeAssignement(const Type::ResolvedType &rightType, const Type::ResolvedType &leftType) 
{
    return std::visit(
        Utils::Overload{
            [&rightType, &leftType](const Type::ResolvedType::Value &leftValue, const Type::ResolvedType::Value &rightValue)
            {
                assert(leftValue.numeric && rightValue.numeric);
                return (rightType == leftType);
            },
            [](const Type::ResolvedType::Pointer &rightPointer, const Type::ResolvedType::Pointer &leftPointer)
            {
                return checkPointerTypeAssignement(*rightPointer.valueType, *leftPointer.valueType);
            },
            // Otherwise, pointers with different levels of indirection e.g. int* and int** are being compared
            [](auto, auto) { return false; }},
        rightType.detail, leftType.detail);
}
//---------------------------------------------------------------------------
bool checkForConstRemoval(const Type::ResolvedType &rightType, const Type::ResolvedType &leftType) 
{
    // If const is being removed
    if (rightType.hasQualifier(Type::Qualifier::CONSTANT) && !leftType.hasQualifier(Type::Qualifier::CONSTANT)) {
        return false;
    }

    return std::visit(
        Utils::Overload{
            // If both are value types
            [](const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Value &leftValue)
            {
                return true;
            },
            // Otherwise, if both are pointers, recurse through value type
            [](const Type::ResolvedType::Pointer &rightPointer, const Type::ResolvedType::Pointer &leftPointer)
            {
                return checkForConstRemoval(*rightPointer.valueType, *leftPointer.valueType);
            },
            // Otherwise, pointers with different levels of indirection e.g. int* and int** are being compared
            [](auto, auto) { return false; }},
        rightType.detail, leftType.detail);
}
//---------------------------------------------------------------------------
bool checkImplicitConversion(const Type::ResolvedType &rightType, const Type::ResolvedType &leftType, Token::Type op = Token::Type::EQUAL)
{
    return std::visit(
        Utils::Overload{
            // If both are numeric, return true as any numeric types can be assigned
            [op](const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Value &leftValue)
            {
                // If operator requires it and both arguments are integers, return true
                assert(leftValue.numeric && rightValue.numeric);
                if (op == Token::Type::PERCENT_EQUAL || op == Token::Type::SHIFT_LEFT_EQUAL
                    || op == Token::Type::SHIFT_RIGHT_EQUAL || op == Token::Type::CARET
                    || op == Token::Type::AMPERSAND_EQUAL || op == Token::Type::PIPE_EQUAL)
                {
                    return (leftValue.numeric->isIntegral && rightValue.numeric->isIntegral);
                }
                // Otherwise, assignement will work for any numeric type
                else {
                    return true;
                }
            },
            // Otherwise, if both are pointers, recurse through value type
            [op, &leftType, &rightType]
            (const Type::ResolvedType::Pointer &rightPointer, const Type::ResolvedType::Pointer &leftPointer)
            {
                // If operator is equals
                if (op == Token::Type::EQUAL) {
                    // Check that value type at the end matches
                    if (!checkPointerTypeAssignement(*rightPointer.valueType, *leftPointer.valueType)) {
                        return false;
                    }
                    // Check we're not trying to maketype less const
                    else if(!checkForConstRemoval(rightType, leftType)) {
                        return false;
                    }
                    else {
                        return true;
                    }
                }
                // Two pointers can only be assigned with =
                else {
                    return false;
                }
            },
            // Otherwise, if left is pointer and right is numeric, 
            [op](const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Pointer &leftPointer) 
            {
                assert(rightValue.numeric);
                if (op == Token::Type::PLUS_EQUAL || op == Token::Type::MINUS_EQUAL) {
                    return rightValue.numeric->isIntegral;
                }
                else {
                    return false;
                }
            },
            // Otherwise, we're trying to assign invalid types
            [](auto, auto) { return false; }},
        rightType.detail, leftType.detail);
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
    virtual void define(const Token &name, const Type::ResolvedType &type, ErrorHandlerBase &errorHandler) final
    {
        if(!m_Types.try_emplace(name.lexeme, type).second) {
            errorHandler.error(name, "Redeclaration of variable");
            throw TypeCheckError();
        }
    }

    virtual std::vector<Type::ResolvedType> getTypes(const Token &name, ErrorHandlerBase &errorHandler) final
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
    std::unordered_map<std::string, Type::ResolvedType> m_Types;
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
        // Evaluate array type
        auto arrayType = evaluateType(arraySubscript.getArray());

        // If pointer is indeed a pointer
        if(arrayType.isPointer()) {
            // Evaluate pointer type
            auto indexType = evaluateType(arraySubscript.getIndex());
            if (!indexType.isNumeric() || !indexType.getNumeric().isIntegral) {
                m_ErrorHandler.error(arraySubscript.getClosingSquareBracket(),
                                     "Invalid subscript index type '" + getDescription(indexType) + "'");
                throw TypeCheckError();
            }

            // Use value type of array
            setExpressionType(&arraySubscript, *arrayType.getPointer().valueType);
        }
        // Otherwise
        else {
            m_ErrorHandler.error(arraySubscript.getClosingSquareBracket(), "Subscripted object is not a pointer");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Assignment &assignment) final
    {
        const auto leftType = evaluateType(assignment.getAssignee());
        const auto rightType = evaluateType(assignment.getValue());

        // If existing type is a const qualified and isn't being initialized, give error
        if(leftType.hasQualifier(Type::Qualifier::CONSTANT)) {
            m_ErrorHandler.error(assignment.getOperator(), "Assignment of read-only variable");
            throw TypeCheckError();
        }
        // Otherwise, if implicit conversion fails, give error
        else if (!checkImplicitConversion(rightType, leftType, assignment.getOperator().type)) {
            m_ErrorHandler.error(assignment.getOperator(), "Invalid operand types '" + getDescription(leftType) + "' and '" + getDescription(rightType));
            throw TypeCheckError();
        }

        setExpressionType(&assignment, leftType);
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

            // Visit permutations of left and right types
            const auto resultType = std::visit(
                Utils::Overload{
                    // If both operands are numeric
                    [&leftType, &rightType, opType, this]
                    (const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Value &leftValue) -> std::optional<Type::ResolvedType>
                    {
                        // If operator requires integer operands
                        if (opType == Token::Type::PERCENT || opType == Token::Type::SHIFT_LEFT
                            || opType == Token::Type::SHIFT_RIGHT || opType == Token::Type::CARET
                            || opType == Token::Type::AMPERSAND || opType == Token::Type::PIPE)
                        {
                            // Check that operands are integers
                            if (leftValue.numeric->isIntegral && rightValue.numeric->isIntegral) {
                                // If operator is a shift, promote left type
                                if (opType == Token::Type::SHIFT_LEFT || opType == Token::Type::SHIFT_RIGHT) {
                                    return Type::getPromotedType(leftType);
                                }
                                // Otherwise, take common type
                                else {
                                    return Type::getCommonType(leftType, rightType);
                                }
                            }
                            else {
                                return std::nullopt;
                            }
                        }
                        // Otherwise, any numeric type will do, take common type
                        else {
                            return Type::getCommonType(leftType, rightType);
                        }
                    },
                    // Otherwise, if both operands are pointers
                    [&binary, &leftType, &rightType, opType, this]
                    (const Type::ResolvedType::Pointer &rightPointer, const Type::ResolvedType::Pointer &leftPointer) -> std::optional<Type::ResolvedType>
                    {
                        // If operator is minus and pointer types match
                        if (opType == Token::Type::MINUS && leftType == rightType) {
                            // **TODO** should be std::ptrdiff/Int64
                            return Type::Int32;
                        }
                        else {
                            return std::nullopt;
                        }
                    },
                    // Otherwise, if right is numeric and left is pointer
                    [&binary, &leftType, &rightType, opType, this]
                    (const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Pointer &leftPointer) -> std::optional<Type::ResolvedType>
                    {
                        // If operator is valid and numeric type is integer
                        // P + n or P - n
                        if ((opType == Token::Type::PLUS || opType == Token::Type::MINUS) && rightValue.numeric->isIntegral) {
                            return leftType;
                        }
                        else {
                             return std::nullopt;
                        }
                    },
                    // Otherwise, if right is pointer and left is numeric
                    [&binary, &rightType, opType, this]
                    (const Type::ResolvedType::Pointer &rightPointer, const Type::ResolvedType::Value &leftValue) -> std::optional<Type::ResolvedType>
                    {
                        // n + P
                        if (opType == Token::Type::PLUS && leftValue.numeric->isIntegral) {
                            return rightType;
                        }
                        else {
                            return std::nullopt;
                        }
                    },
                    // Otherwise, operator is being applied to unsupported types
                    [](auto, auto) -> std::optional<Type::ResolvedType>
                    {
                        return std::nullopt;
                    }},
                rightType.detail, leftType.detail);

                if (resultType) {
                    setExpressionType(&binary, *resultType);
                }
                else {
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + getDescription(leftType) + "' and '" + getDescription(rightType));
                    throw TypeCheckError();
                }
        }
    }

    virtual void visit(const Expression::Call &call) final
    {
        // Evaluate argument types and store in top of stack
        m_CallArguments.emplace();
        std::transform(call.getArguments().cbegin(), call.getArguments().cend(), std::back_inserter(m_CallArguments.top()),
                       [this](const auto &a){ return evaluateType(a.get()); });

        // Evaluate callee type
        auto calleeType = evaluateType(call.getCallee());

        // Pop stack
        m_CallArguments.pop();

        // If callee's a function, type is return type of function
        if (calleeType.isFunction()) {
            setExpressionType(&call, *calleeType.getFunction().returnType);
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
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + getDescription(cast.getType()) + "' and '" + getDescription(rightType));
            throw TypeCheckError();
        }

        const auto resultType = std::visit(
            Utils::Overload{
                // If types are numeric, any cast goes
                [&cast](const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Value &castValue) -> std::optional<Type::ResolvedType>
                {
                    if (rightValue.numeric && castValue.numeric) {
                        return cast.getType();
                    }
                    else {
                        return std::nullopt;
                    }
                },
                // Otherwise, if we're trying to cast pointer to pointer
                [&cast](const Type::ResolvedType::Pointer &rightPointer, const Type::ResolvedType::Pointer &castPointer) -> std::optional<Type::ResolvedType>
                {
                   // Check that value type at the end matches
                    if (checkPointerTypeAssignement(*rightPointer.valueType, *castPointer.valueType)) {
                        return cast.getType();
                    }
                    else {
                        return std::nullopt;
                    }
                },
                // Otherwise, pointers can't be cast to non-pointers and vice versa
                [](auto, auto) -> std::optional<Type::ResolvedType>
                { 
                    return std::nullopt; 
                }},
            rightType.detail, cast.getType().detail);

        if (resultType) {
            setExpressionType(&cast, *resultType);
        }
        else {
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + getDescription(cast.getType()) + "' and '" + getDescription(rightType));
             throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Conditional &conditional) final
    {
        const auto trueType = evaluateType(conditional.getTrue());
        const auto falseType = evaluateType(conditional.getFalse());
        if (trueType.isNumeric() && falseType.isNumeric()) {
            // **TODO** check behaviour
            const auto commonType = Type::getCommonType(trueType, falseType);
            if(trueType.hasQualifier(Type::Qualifier::CONSTANT) || falseType.hasQualifier(Type::Qualifier::CONSTANT)) {
                setExpressionType(&conditional, commonType.addQualifier(Type::Qualifier::CONSTANT));
            }
            else {
                setExpressionType(&conditional, commonType);
            }
        }
        else {
            m_ErrorHandler.error(conditional.getQuestion(),
                                 "Invalid operand types '" + getDescription(trueType) + "' and '" + getDescription(falseType) + "' to conditional");
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
            setExpressionType(&literal, Type::Double);
        }
        else if (literal.getValue().type == Token::Type::FLOAT_NUMBER) {
            setExpressionType(&literal, Type::Float);
        }
        else if (literal.getValue().type == Token::Type::SCALAR_NUMBER) {
            // **TODO** cache
            assert(false);
            // **THINK** why not resolve here?
            //setExpressionType(&literal, new Type::NumericTypedef("scalar"));
        }
        else if (literal.getValue().type == Token::Type::INT32_NUMBER) {
            setExpressionType(&literal, Type::Int32);
        }
        else if (literal.getValue().type == Token::Type::UINT32_NUMBER) {
            setExpressionType(&literal, Type::Uint32);
        }
        else if(literal.getValue().type == Token::Type::STRING) {
            setExpressionType(&literal, Type::ResolvedType::createPointer(Type::Int8, Type::Qualifier::CONSTANT));
        }
        else {
            assert(false);
        }
    }

    virtual void visit(const Expression::Logical &logical) final
    {
        logical.getLeft()->accept(*this);
        logical.getRight()->accept(*this);
        setExpressionType(&logical, Type::Int32);
    }

    virtual void visit(const Expression::PostfixIncDec &postfixIncDec) final
    {
        // **TODO** more general lvalue thing
        const auto lhsType = evaluateType(postfixIncDec.getTarget());
        if(lhsType.hasQualifier(Type::Qualifier::CONSTANT)) {
            m_ErrorHandler.error(postfixIncDec.getOperator(), "Increment/decrement of read-only variable");
            throw TypeCheckError();
        }
        else {
            setExpressionType(&postfixIncDec, lhsType);
        }
    }

    virtual void visit(const Expression::PrefixIncDec &prefixIncDec) final
    {
        // **TODO** more general lvalue thing
        const auto rhsType = evaluateType(prefixIncDec.getTarget());
         if(rhsType.hasQualifier(Type::Qualifier::CONSTANT)) {
            m_ErrorHandler.error(prefixIncDec.getOperator(), "Increment/decrement of read-only variable");
            throw TypeCheckError();
        }
        else {
            setExpressionType(&prefixIncDec, rhsType);
        }
    }

    virtual void visit(const Expression::Variable &variable)
    {
        // If type is unambiguous and not a function
        const auto varTypes = m_Environment.get().getTypes(variable.getName(), m_ErrorHandler);
        if (varTypes.size() == 1 && !varTypes.front().isFunction()) {
            setExpressionType(&variable, varTypes.front());
        }
        // Otherwise
        else {
            // Check that there are call arguments on the stack
            assert(!m_CallArguments.empty());

            // Loop through variable types
            std::vector<std::pair<Type::ResolvedType, std::vector<int>>> viableFunctions;
            for(const auto &type : varTypes) {
                // If  function is non-variadic and number of arguments match
                const auto &argumentTypes = type.getFunction().argTypes;
                if(m_CallArguments.top().size() == argumentTypes.size()) {
                    // Create vector to hold argument conversion rank
                    std::vector<int> argumentConversionRank;
                    argumentConversionRank.reserve(m_CallArguments.top().size());

                    // Loop through arguments
                    bool viable = true;
                    auto c = m_CallArguments.top().cbegin();
                    auto a = argumentTypes.cbegin();
                    for(;c != m_CallArguments.top().cend(); c++, a++) {
                        const auto argConversionRank = std::visit(
                            Utils::Overload{
                                // If types are numeric, any cast goes
                                [c, a](const Type::ResolvedType::Value &cValue, const Type::ResolvedType::Value &aValue) -> std::optional<int>
                                {
                                    // If names are identical, match is exact
                                    // **TODO** we don't care about qualifiers
                                    if(*c == *a) {
                                        return 0;
                                    }
                                    // Integer promotion
                                    else if(*a == Type::Int32 && cValue.numeric->isIntegral
                                            && cValue.numeric->rank < Type::Int32.getNumeric().rank)
                                    {
                                        return 1;
                                    }
                                    // Float promotion
                                    else if(*a == Type::Double && *c == Type::Float) {
                                        return 1;
                                    }
                                    // Otherwise, numeric conversion
                                    // **TODO** integer to scalar promotion should be lower ranked than general conversion
                                    else {
                                        return 2;
                                    }
                                },
                                // Otherwise, if we're trying to cast pointer to pointer
                                [](const Type::ResolvedType::Pointer &cPointer, const Type::ResolvedType::Pointer &aPointer) -> std::optional<int>
                                {
                                    // Check that value type at the end matches
                                    if (checkPointerTypeAssignement(*cPointer.valueType, *aPointer.valueType)) {
                                        return 0;
                                    } 
                                    else {
                                        return std::nullopt;
                                    }
                                },
                                // Otherwise, pointers can't be cast to non-pointers and vice versa
                                [](auto, auto) -> std::optional<int>
                                { 
                                    return std::nullopt; 
                                }},
                            c->detail, a->detail);

                        // If there is a valid conversion between argument and definition
                        if (argConversionRank) {
                            argumentConversionRank.push_back(*argConversionRank);
                        }
                        // Otherwise, this function is not viable
                        else {
                            viable = false;
                        }
                    }

                    // If function is viable, add to vector along with vector of conversion ranks
                    if(viable) {
                        viableFunctions.emplace_back(type, argumentConversionRank);
                    }
                }
            }

            // If there are no viable candidates, give error
            if(viableFunctions.empty()) {
                m_ErrorHandler.error(variable.getName(),
                                        "No viable function candidates for '" + variable.getName().lexeme + "'");
                throw TypeCheckError();
            }
            // Otherwise, sort lexigraphically by conversion rank and return type of lowest
            // **TODO** handle case when best is ambiguous
            else {
                std::sort(viableFunctions.begin(), viableFunctions.end(),
                          [](auto &f1, auto &f2){ return (f1.second < f2.second); });
                setExpressionType(&variable, viableFunctions.front().first);
            }
        }
    }

    virtual void visit(const Expression::Unary &unary) final
    {
        const auto rightType = evaluateType(unary.getRight());

        // If operator is pointer de-reference
        if (unary.getOperator().type == Token::Type::STAR) {
            if (rightType.isPointer()) {
                 setExpressionType(&unary, *rightType.getPointer().valueType);
            }
            else {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + getDescription(rightType) + "'");
                throw TypeCheckError();
            }
        }
        // Otherwise
        else if (rightType.isNumeric()) {
            // If operator is arithmetic, return promoted type
            if (unary.getOperator().type == Token::Type::PLUS || unary.getOperator().type == Token::Type::MINUS) {
                // **THINK** const through these?
                setExpressionType(&unary, Type::getPromotedType(rightType));
            }
            // Otherwise, if operator is bitwise
            else if (unary.getOperator().type == Token::Type::TILDA) {
                // If type is integer, return promoted type
                if (rightType.getNumeric().isIntegral) {
                    // **THINK** const through these?
                    setExpressionType(&unary, Type::getPromotedType(rightType));
                }
                else {
                    m_ErrorHandler.error(unary.getOperator(),
                                            "Invalid operand type '" + getDescription(rightType) + "'");
                    throw TypeCheckError();
                }
            }
            // Otherwise, if operator is logical
            else if (unary.getOperator().type == Token::Type::NOT) {
                setExpressionType(&unary, Type::Int32);
            }
            // Otherwise, if operator is address of, return pointer type
            else if (unary.getOperator().type == Token::Type::AMPERSAND) {
                setExpressionType(&unary, Type::ResolvedType::createPointer(rightType));
            }
        }
        else {
            m_ErrorHandler.error(unary.getOperator(),
                                    "Invalid operand type '" + getDescription(rightType) + "'");
            throw TypeCheckError();
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
            if (!valType.isNumeric() || !valType.getNumeric().isIntegral) {
                m_ErrorHandler.error(labelled.getKeyword(),
                                     "Invalid case value '" + getDescription(valType) + "'");
                throw TypeCheckError();
            }
        }

        labelled.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Switch &switchStatement) final
    {
        auto condType = evaluateType(switchStatement.getCondition());
        if (!condType.isNumeric() || !condType.getNumeric().isIntegral) {
            m_ErrorHandler.error(switchStatement.getSwitch(),
                                 "Invalid condition '" + getDescription(condType) + "'");
            throw TypeCheckError();
        }

        m_InSwitch = true;
        switchStatement.getBody()->accept(*this);
        m_InSwitch = false;
    }

    virtual void visit(const Statement::VarDeclaration &varDeclaration) final
    {
        const auto decType = varDeclaration.getType();
        for (const auto &var : varDeclaration.getInitDeclaratorList()) {
            m_Environment.get().define(std::get<0>(var), decType, m_ErrorHandler);

            // If variable has an initialiser expression, check that 
            // it can be implicitly converted to variable type
            if (std::get<1>(var)) {
                const auto initialiserType = evaluateType(std::get<1>(var).get());
                if (!checkImplicitConversion(initialiserType, decType)) {
                    m_ErrorHandler.error(std::get<0>(var), "Invalid operand types '" + getDescription(decType) + "' and '" + getDescription(initialiserType));
                }
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
    Type::ResolvedType evaluateType(const Expression::Base *expression)
    {
        expression->accept(*this);
        return m_ResolvedTypes.at(expression);
    }
   
    void setExpressionType(const Expression::Base *expression, const Type::ResolvedType &type)
    {
        if (!m_ResolvedTypes.emplace(expression, type).second) {
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
    std::stack<std::vector<Type::ResolvedType>> m_CallArguments;
    bool m_InLoop;
    bool m_InSwitch;
};
}   // Anonymous namespace

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker::EnvironmentBase
//---------------------------------------------------------------------------
Type::ResolvedType EnvironmentBase::getType(const Token &name, ErrorHandlerBase &errorHandler)
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
// GeNN::Transpiler::TypeChecker
//---------------------------------------------------------------------------
ResolvedTypeMap GeNN::Transpiler::TypeChecker::typeCheck(const Statement::StatementList &statements, EnvironmentBase &environment, 
                                                         const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    ResolvedTypeMap expressionTypes;
    EnvironmentInternal internalEnvironment(environment);
    Visitor visitor(statements, context, internalEnvironment, expressionTypes, errorHandler);
    return expressionTypes;
}
//---------------------------------------------------------------------------
Type::ResolvedType GeNN::Transpiler::TypeChecker::typeCheck(const Expression::Base *expression, EnvironmentBase &environment,
                                                    const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    ResolvedTypeMap expressionTypes;
    EnvironmentInternal internalEnvironment(environment);
    Visitor visitor(expression, context, internalEnvironment, expressionTypes, errorHandler);
    return expressionTypes.at(expression);
}
