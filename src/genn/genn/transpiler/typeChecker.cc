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
bool checkPointerTypeAssignement(const Type::ResolvedType &rightType, const Type::ResolvedType &leftType) 
{
    return std::visit(
        Utils::Overload{
            [](const Type::ResolvedType::Value &leftValue, const Type::ResolvedType::Value &rightValue)
            {
                return (rightValue == leftValue);
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
    if (rightType.isConst && !leftType.isConst) {
        return false;
    }

    return std::visit(
        Utils::Overload{
            // If both are value types
            [](const Type::ResolvedType::Value&, const Type::ResolvedType::Value&)
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
            [op](const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Pointer&)
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
// Visitor
//---------------------------------------------------------------------------
class Visitor : public Expression::Visitor<>, public Statement::Visitor<>
{
public:
    Visitor(const Statement::StatementList<> &statements, EnvironmentInternal &environment, 
            const Type::TypeContext &context, ErrorHandlerBase &errorHandler, 
            StatementHandler forEachSynapseHandler)
    :   Visitor(environment, context, errorHandler, forEachSynapseHandler)
    {
        for (auto &s : statements) {
            s.get()->accept(*this);
        }
    }
    
    Visitor(const Expression::Base<> *expression, EnvironmentInternal &environment, 
            const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
    :   Visitor(environment, context, errorHandler, nullptr)
    {
        expression->accept(*this);
    }
    
private:
    Visitor(EnvironmentInternal &environment, const Type::TypeContext &context, 
            ErrorHandlerBase &errorHandler, StatementHandler forEachSynapseHandler)
    :   m_Environment(environment), m_Context(context), m_ErrorHandler(errorHandler), 
        m_ForEachSynapseHandler(forEachSynapseHandler)
    {
    }

    //---------------------------------------------------------------------------
    // Expression::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Expression::ArraySubscript<> &arraySubscript) final
    {
        // Evaluate index type
        auto indexType = evaluateType(arraySubscript.getIndex());
        if (!indexType.isNumeric() || !indexType.getNumeric().isIntegral || indexType.getValue().isWriteOnly) {
            m_ErrorHandler.error(arraySubscript.getClosingSquareBracket(),
                                 "Invalid subscript index type '" + indexType.getName() + "'");
            throw TypeCheckError();
        }

        // Place index type on top of stack so it can be used
        // to type check array subscript override function
        m_CallArguments.emplace();
        m_CallArguments.top().reserve(1);
        m_CallArguments.top().push_back(indexType);

        // Evaluate array type
        auto arrayType = evaluateType(arraySubscript.getArray());
        
        // Pop stack
        m_CallArguments.pop();

        // If array type is a pointer, expression type is value type
        if(arrayType.isPointer()) {
            setExpressionType(&arraySubscript, *arrayType.getPointer().valueType);
        }
        // Otherwise, if 'array' is a function with the array subscript override, use function return type
        else if(arrayType.isFunction() 
                && arrayType.getFunction().hasFlag(Type::FunctionFlags::ARRAY_SUBSCRIPT_OVERRIDE))
        {
            setExpressionType(&arraySubscript, *arrayType.getFunction().returnType);
        }
        // Otherwise
        else {
            m_ErrorHandler.error(arraySubscript.getClosingSquareBracket(), "Subscripted object is not a pointer");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Assignment<> &assignment) final
    {
        const auto leftType = evaluateType(assignment.getAssignee());
        const auto rightType = evaluateType(assignment.getValue());

        // If existing type is a const qualified and isn't being initialized, give error
        if(leftType.isConst) {
            m_ErrorHandler.error(assignment.getOperator(), "Assignment of read-only variable");
            throw TypeCheckError();
        }
        // If LHS is write-only and the operator isn't a plain assignement
        else if(leftType.isValue() && leftType.getValue().isWriteOnly && assignment.getOperator().type != Token::Type::EQUAL) {
            m_ErrorHandler.error(assignment.getOperator(), "Invalid operator for assignement of write-only variable");
            throw TypeCheckError();
        }
        // If RHS is write-only, cannot be assigned
        else if(rightType.isValue() && rightType.getValue().isWriteOnly) {
            m_ErrorHandler.error(assignment.getOperator(), "Invalid operand type '" + rightType.getName() + "'");
            throw TypeCheckError();
        }
        // Otherwise, if implicit conversion fails, give error
        else if (!checkImplicitConversion(rightType, leftType, assignment.getOperator().type)) {
            m_ErrorHandler.error(assignment.getOperator(), "Invalid operand types '" + leftType.getName() + "' and '" + rightType.getName() + "'");
            throw TypeCheckError();
        }

        setExpressionType(&assignment, leftType);
    }

    virtual void visit(const Expression::Binary<> &binary) final
    {
        const auto opType = binary.getOperator().type;
        const auto leftType = evaluateType(binary.getLeft());
        const auto rightType = evaluateType(binary.getRight());
        if (opType == Token::Type::COMMA) {
            setExpressionType(&binary, rightType);
        }
        else {
            // Visit permutations of left and right types
            const auto resultType = std::visit(
                Utils::Overload{
                    // If both operands are numeric
                    [&leftType, &rightType, opType]
                    (const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Value &leftValue) -> std::optional<Type::ResolvedType>
                    {
                        // If either type is write only or non-numeric, error
                        if(leftValue.isWriteOnly || rightValue.isWriteOnly 
                           || !leftType.isNumeric() || !rightType.isNumeric()) 
                        {
                            return std::nullopt;
                        }

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
                    // **TODO** don't pointer types need to be the same?
                    [&leftType, &rightType, opType]
                    (const Type::ResolvedType::Pointer&, const Type::ResolvedType::Pointer&) -> std::optional<Type::ResolvedType>
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
                    [&leftType, opType]
                    (const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Pointer&) -> std::optional<Type::ResolvedType>
                    {
                        // If right type is write only or non-numeric, error
                        if(rightValue.isWriteOnly || !rightValue.numeric) {
                            return std::nullopt;
                        }
                        // If operator is valid and numeric type is integer
                        // P + n or P - n
                        else if ((opType == Token::Type::PLUS || opType == Token::Type::MINUS) && rightValue.numeric->isIntegral) {
                            return leftType;
                        }
                        else {
                             return std::nullopt;
                        }
                    },
                    // Otherwise, if right is pointer and left is numeric
                    [&rightType, opType]
                    (const Type::ResolvedType::Pointer&, const Type::ResolvedType::Value &leftValue) -> std::optional<Type::ResolvedType>
                    {
                        // If left type is write only or non-numeric, error
                        if(leftValue.isWriteOnly || !leftValue.numeric) {
                            return std::nullopt;
                        }
                        // n + P
                        else if (opType == Token::Type::PLUS && leftValue.numeric->isIntegral) {
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
                    m_ErrorHandler.error(binary.getOperator(), "Invalid operand types '" + leftType.getName() + "' and '" + rightType.getName());
                    throw TypeCheckError();
                }
        }
    }

    virtual void visit(const Expression::Call<> &call) final
    {
        // Evaluate argument types and store in top of stack
        std::vector<Type::ResolvedType> arguments;
        arguments.reserve(call.getArguments().size());
        for(const auto &arg : call.getArguments()) {
            const auto argType = evaluateType(arg.get());
            
            // If argument is write-only, throw
            if(argType.isValue() && argType.getValue().isWriteOnly) {
                m_ErrorHandler.error(call.getClosingParen(), "Write-only argument type '" + argType.getName() + "'");
                throw TypeCheckError();
            }
            arguments.push_back(argType);
        }
        
        // Push arguments onto stack
        m_CallArguments.push(arguments);

        // Evaluate callee type
        auto calleeType = evaluateType(call.getCallee());

        // Pop stack
        m_CallArguments.pop();

        // If callee's a function without the array subscript override f, type is return type of function
        if (calleeType.isFunction() 
            && !calleeType.getFunction().hasFlag(Type::FunctionFlags::ARRAY_SUBSCRIPT_OVERRIDE)) 
        {
            setExpressionType(&call, *calleeType.getFunction().returnType);
        }
        // Otherwise
        else {
            m_ErrorHandler.error(call.getClosingParen(), "Called object is not a function");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Cast<> &cast) final
    {
        // Evaluate type of expression we're casting
        const auto rightType = evaluateType(cast.getExpression());

        const auto resultType = std::visit(
            Utils::Overload{
                // If types are numeric, any cast goes
                [&cast](const Type::ResolvedType::Value &rightValue, const Type::ResolvedType::Value &castValue) -> std::optional<Type::ResolvedType>
                {
                    if (rightValue.numeric && castValue.numeric && !rightValue.isWriteOnly) {
                        return cast.getType();
                    }
                    else {
                        return std::nullopt;
                    }
                },
                // Otherwise, if we're trying to cast pointer to pointer
                [&cast, &rightType](const Type::ResolvedType::Pointer &rightPointer, const Type::ResolvedType::Pointer &castPointer) -> std::optional<Type::ResolvedType>
                {
                    // Check that value type at the end matches
                    if (!checkPointerTypeAssignement(*rightPointer.valueType, *castPointer.valueType)) {
                        return std::nullopt;
                    }
                    // Check we're not trying to maketype less const
                    else if(!checkForConstRemoval(rightType, cast.getType())) {
                        return std::nullopt;
                    }
                    else {
                        return cast.getType();
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
            m_ErrorHandler.error(cast.getClosingParen(), "Invalid operand types '" + cast.getType().getName() + "' and '" + rightType.getName());
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Conditional<> &conditional) final
    {
        const auto conditionType = evaluateType(conditional.getCondition());
        if(!conditionType.isScalar()) {
            m_ErrorHandler.error(conditional.getQuestion(), "Invalid condition type '" + conditionType.getName() + "'");
            throw TypeCheckError();
        }
        
        const auto trueType = evaluateType(conditional.getTrue());
        const auto falseType = evaluateType(conditional.getFalse());
        if (trueType.isNumeric() && falseType.isNumeric()) {
            // **TODO** check behaviour
            const auto commonType = Type::getCommonType(trueType, falseType);
            if(trueType.isConst || falseType.isConst) {
                setExpressionType(&conditional, commonType.addConst());
            }
            else {
                setExpressionType(&conditional, commonType);
            }
        }
        else {
            m_ErrorHandler.error(conditional.getQuestion(),
                                 "Invalid operand types '" + trueType.getName() + "' and '" + falseType.getName() + "' to conditional");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::Grouping<> &grouping) final
    {
        const auto type = evaluateType(grouping.getExpression());
        setAnnotatedExpression<Expression::Grouping<TypeAnnotation>>(std::move(m_AnnotatedExpression), type);
    }

    virtual void visit(const Expression::Literal<> &literal) final
    {
        // Convert literal token type to type
        if (literal.getValue().type == Token::Type::DOUBLE_NUMBER) {
            setAnnotatedExpression<Expression::Literal<TypeAnnotation>>(literal.getValue(), Type::Double);
        }
        else if (literal.getValue().type == Token::Type::FLOAT_NUMBER) {
            setAnnotatedExpression<Expression::Literal<TypeAnnotation>>(literal.getValue(), Type::Float);
        }
        else if (literal.getValue().type == Token::Type::SCALAR_NUMBER) {
            setAnnotatedExpression<Expression::Literal<TypeAnnotation>>(literal.getValue(), m_Context.at("scalar"));
        }
        else if (literal.getValue().type == Token::Type::INT32_NUMBER) {
            setAnnotatedExpression<Expression::Literal<TypeAnnotation>>(literal.getValue(), Type::Int32);
        }
        else if (literal.getValue().type == Token::Type::UINT32_NUMBER) {
            setAnnotatedExpression<Expression::Literal<TypeAnnotation>>(literal.getValue(), Type::Uint32);
        }
        else if(literal.getValue().type == Token::Type::BOOLEAN) {
            setAnnotatedExpression<Expression::Literal<TypeAnnotation>>(literal.getValue(), Type::Bool);
        }
        else if(literal.getValue().type == Token::Type::STRING) {
            setAnnotatedExpression<Expression::Literal<TypeAnnotation>>(literal.getValue(), Type::Int8.createPointer(true));
        }
        else {
            assert(false);
        }
    }

    virtual void visit(const Expression::Logical<> &logical) final
    {
        auto left = annotateExpression(logical.getLeft());
        const auto &leftType = left->getType();

        auto right = annotateExpression(logical.getRight());
        const auto &rightType = right->getType();

        if(leftType.isScalar() && rightType.isScalar()) {
            setAnnotatedExpression<Expression::Logical<TypeAnnotation>>(std::move(left),
                                                                        logical.getOperator(),
                                                                        std::move(right),
                                                                        Type::Int32);
        }
        else {
            m_ErrorHandler.error(logical.getOperator(), "Invalid operand types '" + leftType.getName() + "' and '" + rightType.getName());
            throw TypeCheckError();
        }
    }

    virtual void visit(const Expression::PostfixIncDec<> &postfixIncDec) final
    {
        // **TODO** more general lvalue thing
        auto lhs = annotateExpression(postfixIncDec.getTarget());
        const auto &lhsType = lhs->getType();
        if(lhsType.isConst || !lhsType.isScalar()) {
            m_ErrorHandler.error(postfixIncDec.getOperator(),"Invalid operand type '" + lhsType.getName() + "'");
            throw TypeCheckError();
        }
        else {
            setAnnotatedExpression<Expression::PostfixIncDec<TypeAnnotation>>(postfixIncDec.getOperator(),
                                                                              std::move(lhs), lhsType);
        }
    }

    virtual void visit(const Expression::PrefixIncDec<> &prefixIncDec) final
    {
        // **TODO** more general lvalue thing
        auto rhs = annotateExpression(prefixIncDec.getTarget());
        const auto &rhsType = rhs->getType();
        if(rhsType.isConst || !rhsType.isScalar()) {
            m_ErrorHandler.error(prefixIncDec.getOperator(),"Invalid operand type '" + rhsType.getName() + "'");
            throw TypeCheckError();
        }
        else {
            setAnnotatedExpression<Expression::PrefixIncDec<TypeAnnotation>>(prefixIncDec.getOperator(),
                                                                             std::move(rhs), rhsType);
        }
    }

    virtual void visit(const Expression::Identifier<> &variable)
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
                // If  function is non-variadic and number of arguments 
                // match or variadic and enough arguments are provided
                const auto &argumentTypes = type.getFunction().argTypes;
                const bool variadic = type.getFunction().hasFlag(Type::FunctionFlags::VARIADIC);
                if((!variadic && m_CallArguments.top().size() == argumentTypes.size())
                   || (variadic && m_CallArguments.top().size() >= argumentTypes.size()))
                {
                    // Create vector to hold argument conversion rank
                    std::vector<int> argumentConversionRank;
                    argumentConversionRank.reserve(m_CallArguments.top().size());

                    // Loop through arguments
                    // **NOTE** we loop through function TYPE arguments to avoid variadic
                    bool viable = true;
                    auto c = m_CallArguments.top().cbegin();
                    auto a = argumentTypes.cbegin();
                    for(;a != argumentTypes.cend(); c++, a++) {
                        const auto argConversionRank = std::visit(
                            Utils::Overload{
                                // If types are numeric, any cast goes
                                [c, a](const Type::ResolvedType::Value &cValue, const Type::ResolvedType::Value&) -> std::optional<int>
                                {
                                    // If types are identical, match is exact
                                    const auto unqualifiedA = a->removeConst();
                                    const auto unqualifiedC = c->removeConst();
                                    if(unqualifiedC == unqualifiedA) {
                                        return 0;
                                    }
                                    // Integer promotion
                                    else if(unqualifiedA == Type::Int32 && cValue.numeric->isIntegral
                                            && cValue.numeric->rank < Type::Int32.getNumeric().rank)
                                    {
                                        return 1;
                                    }
                                    // Float promotion
                                    else if(unqualifiedA == Type::Double && unqualifiedC == Type::Float) {
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

    virtual void visit(const Expression::Unary<> &unary) final
    {
        auto right = annotateExpression(unary.getRight());
        const auto &rightType = right->getType();

        // If operator is pointer de-reference
         if (unary.getOperator().type == Token::Type::STAR) {
            if (rightType.isPointer()) {
                setAnnotatedExpression<Expression::Unary<TypeAnnotation>>(unary.getOperator(), std::move(right), 
                                                                          *rightType.getPointer().valueType);
            }
            else {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType.getName() + "'");
                throw TypeCheckError();
            }
        }
        // Otherwise
        else if (rightType.isNumeric() && !rightType.getValue().isWriteOnly) {
            // If operator is arithmetic, return promoted type
            if (unary.getOperator().type == Token::Type::PLUS || unary.getOperator().type == Token::Type::MINUS) {
                // **THINK** const through these?
                setAnnotatedExpression<Expression::Unary<TypeAnnotation>>(unary.getOperator(), std::move(right), 
                                                                          Type::getPromotedType(rightType));
            }
            // Otherwise, if operator is bitwise
            else if (unary.getOperator().type == Token::Type::TILDA) {
                // If type is integer, return promoted type
                if (rightType.getNumeric().isIntegral) {
                    // **THINK** const through these?
                    setAnnotatedExpression<Expression::Unary<TypeAnnotation>>(unary.getOperator(), std::move(right), 
                                                                              Type::getPromotedType(rightType));
                }
                else {
                    m_ErrorHandler.error(unary.getOperator(),
                                            "Invalid operand type '" + rightType.getName() + "'");
                    throw TypeCheckError();
                }
            }
            // Otherwise, if operator is logical
            else if (unary.getOperator().type == Token::Type::NOT) {
                setAnnotatedExpression<Expression::Unary<TypeAnnotation>>(unary.getOperator(), std::move(right), 
                                                                          Type::Int32);
            }
            else {
                m_ErrorHandler.error(unary.getOperator(),
                                     "Invalid operand type '" + rightType.getName() + "'");
                throw TypeCheckError();
            }
        }
        else {
            m_ErrorHandler.error(unary.getOperator(),
                                    "Invalid operand type '" + rightType.getName() + "'");
            throw TypeCheckError();
        }
    }

    //---------------------------------------------------------------------------
    // Statement::Visitor virtuals
    //---------------------------------------------------------------------------
    virtual void visit(const Statement::Break<> &breakStatement) final
    {
        if (m_ActiveLoopStatements.empty() && m_ActiveSwitchStatements.empty()) {
            m_ErrorHandler.error(breakStatement.getToken(), "Statement not within loop");
            throw TypeCheckError(); 
        }
    }

    virtual void visit(const Statement::Compound<> &compound) final
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

    virtual void visit(const Statement::Continue<> &continueStatement) final
    {
        if (m_ActiveLoopStatements.empty()) {
            m_ErrorHandler.error(continueStatement.getToken(), "Statement not within loop");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Statement::Do<> &doStatement) final
    {
        m_ActiveLoopStatements.emplace(&doStatement);
        doStatement.getBody()->accept(*this);
        assert(m_ActiveLoopStatements.top() == &doStatement);
        m_ActiveLoopStatements.pop();
        
        const auto conditionType = evaluateType(doStatement.getCondition());
        if(!conditionType.isScalar()) {
            m_ErrorHandler.error(doStatement.getWhile(), "Invalid condition expression type '" + conditionType.getName() + "'");
            throw TypeCheckError();
        }
    }

    virtual void visit(const Statement::Expression<> &expression) final
    {
        if(expression.getExpression()) {
            expression.getExpression()->accept(*this);
        }
    }

    virtual void visit(const Statement::For<> &forStatement) final
    {
        // Cache reference to current environment
        std::reference_wrapper<EnvironmentInternal> oldEnvironment = m_Environment; 
        
        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;

        // Interpret initialiser if statement present
        if (forStatement.getInitialiser()) {
            forStatement.getInitialiser()->accept(*this);
        }

        if (forStatement.getCondition()) {
            const auto conditionType = evaluateType(forStatement.getCondition());
            if(!conditionType.isScalar()) {
                m_ErrorHandler.error(forStatement.getFor(), "Invalid condition expression type '" + conditionType.getName() + "'");
                throw TypeCheckError();
            }
        }

        if (forStatement.getIncrement()) {
            const auto incrementType = evaluateType(forStatement.getIncrement());
            if(!incrementType.isScalar()) {
                m_ErrorHandler.error(forStatement.getFor(), "Invalid increment expression type '" + incrementType.getName() + "'");
                throw TypeCheckError();
            }
        }

        m_ActiveLoopStatements.emplace(&forStatement);
        forStatement.getBody()->accept(*this);
        assert(m_ActiveLoopStatements.top() == &forStatement);
        m_ActiveLoopStatements.pop();

        // Restore old environment
        m_Environment = oldEnvironment;
    }

    virtual void visit(const Statement::ForEachSynapse<> &forEachSynapseStatement) final
    {
        if(!m_ForEachSynapseHandler) {
            m_ErrorHandler.error(forEachSynapseStatement.getForEachSynapse(), 
                                 "Not supported in this context");
            throw TypeCheckError();
        }
        // Cache reference to current environment
        std::reference_wrapper<EnvironmentInternal> oldEnvironment = m_Environment; 
        
        // Create new environment and set to current
        EnvironmentInternal environment(m_Environment);
        m_Environment = environment;

        // Call handler to define anything required in environment
        m_ForEachSynapseHandler(m_Environment, m_ErrorHandler);

        m_ActiveLoopStatements.emplace(&forEachSynapseStatement);
        forEachSynapseStatement.getBody()->accept(*this);
        assert(m_ActiveLoopStatements.top() == &forEachSynapseStatement);
        m_ActiveLoopStatements.pop();

        // Restore old environment
        m_Environment = oldEnvironment;
    }

    virtual void visit(const Statement::If<> &ifStatement) final
    {
        const auto conditionType = evaluateType(ifStatement.getCondition());
        if(!conditionType.isScalar()) {
            m_ErrorHandler.error(ifStatement.getIf(), "Invalid condition expression type '" + conditionType.getName() + "'");
            throw TypeCheckError();
        }

        ifStatement.getThenBranch()->accept(*this);
        if (ifStatement.getElseBranch()) {
            ifStatement.getElseBranch()->accept(*this);
        }
    }

    virtual void visit(const Statement::Labelled<> &labelled) final
    {
        if (m_ActiveSwitchStatements.empty()) {
            m_ErrorHandler.error(labelled.getKeyword(), "Statement not within switch statement");
            throw TypeCheckError();
        }

        if (labelled.getValue()) {
            auto valType = evaluateType(labelled.getValue());
            if (!valType.isNumeric() || !valType.getNumeric().isIntegral) {
                m_ErrorHandler.error(labelled.getKeyword(),
                                     "Invalid case value '" + valType.getName() + "'");
                throw TypeCheckError();
            }
        }

        labelled.getBody()->accept(*this);
    }

    virtual void visit(const Statement::Switch<> &switchStatement) final
    {
        auto condType = evaluateType(switchStatement.getCondition());
        if (!condType.isNumeric() || !condType.getNumeric().isIntegral || condType.getValue().isWriteOnly) {
            m_ErrorHandler.error(switchStatement.getSwitch(),
                                 "Invalid condition '" + condType.getName() + "'");
            throw TypeCheckError();
        }

        m_ActiveSwitchStatements.emplace(&switchStatement);
        switchStatement.getBody()->accept(*this);
        assert(m_ActiveSwitchStatements.top() == &switchStatement);
        m_ActiveSwitchStatements.pop();
    }

    virtual void visit(const Statement::VarDeclaration<> &varDeclaration) final
    {
        const auto decType = varDeclaration.getType();
        for (const auto &var : varDeclaration.getInitDeclaratorList()) {
            m_Environment.get().define(std::get<0>(var), decType, m_ErrorHandler);

            // If variable has an initialiser expression, check that 
            // it can be implicitly converted to variable type
            if (std::get<1>(var)) {
                const auto initialiserType = evaluateType(std::get<1>(var).get());
                if (!checkImplicitConversion(initialiserType, decType)) {
                    m_ErrorHandler.error(std::get<0>(var), "Invalid operand types '" + decType.getName() + "' and '" + initialiserType.getName());
                    throw TypeCheckError();
                }
            }
        }
    }

    virtual void visit(const Statement::While<> &whileStatement) final
    {
        const auto conditionType = evaluateType(whileStatement.getCondition());
        if(!conditionType.isScalar()) {
            m_ErrorHandler.error(whileStatement.getWhile(), "Invalid condition expression type '" + conditionType.getName() + "'");
            throw TypeCheckError();
        }

        m_ActiveLoopStatements.emplace(&whileStatement);
        whileStatement.getBody()->accept(*this);
        assert(m_ActiveLoopStatements.top() == &whileStatement);
        m_ActiveLoopStatements.pop();
    }

private:
    //---------------------------------------------------------------------------
    // Private methods
    //---------------------------------------------------------------------------
    Expression::ExpressionPtr<TypeAnnotation> annotateExpression(const Expression::Base<> *expression)
    {
        expression->accept(*this);
        return std::move(m_AnnotatedExpression);
    }

    template<typename T, typename... ExpressionArgs>
    void setAnnotatedExpression(ExpressionArgs&&... expressionArgs)
    {
        m_AnnotatedExpression = std::make_unique<T>(std::forward<AnnotationArgs>(expressionArgs)...);
    }
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::reference_wrapper<EnvironmentInternal> m_Environment;
    const Type::TypeContext &m_Context;
    ErrorHandlerBase &m_ErrorHandler;
    StatementHandler m_ForEachSynapseHandler;
    std::stack<std::vector<Type::ResolvedType>> m_CallArguments;
    std::stack<const Statement::Base<>*> m_ActiveLoopStatements;
    std::stack<const Statement::Base<>*> m_ActiveSwitchStatements;
    Expression::ExpressionPtr<TypeAnnotation> m_AnnotatedExpression;
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
// GeNN::Transpiler::TypeChecker::EnvironmentInternal
//---------------------------------------------------------------------------
void EnvironmentInternal::define(const Token &name, const Type::ResolvedType &type, ErrorHandlerBase &errorHandler)
{
    if(!m_Types.try_emplace(name.lexeme, type).second) {
        errorHandler.error(name, "Redeclaration of variable");
        throw TypeCheckError();
    }
}
//---------------------------------------------------------------------------
std::vector<Type::ResolvedType> EnvironmentInternal::getTypes(const Token &name, ErrorHandlerBase &errorHandler)
{
    auto type = m_Types.find(name.lexeme);
    if(type == m_Types.end()) {
        return m_Enclosing.getTypes(name, errorHandler);
    }
    else {
        return {type->second};
    }
}

//---------------------------------------------------------------------------
// GeNN::Transpiler::TypeChecker
//---------------------------------------------------------------------------
Statement::StatementList<TypeAnnotation> GeNN::Transpiler::TypeChecker::typeCheck(
    const Statement::StatementList<> &statements, EnvironmentInternal &environment, 
    const Type::TypeContext &context, ErrorHandlerBase &errorHandler,
    StatementHandler forEachSynapseHandler)
{
    ResolvedTypeMap expressionTypes;
    Visitor visitor(statements, environment, context, errorHandler, forEachSynapseHandler);
    return expressionTypes;
}
//---------------------------------------------------------------------------
Expression::ExpressionPtr<TypeAnnotation> GeNN::Transpiler::TypeChecker::typeCheck(
    const Expression::Base<> *expression, EnvironmentInternal &environment,
    const Type::TypeContext &context, ErrorHandlerBase &errorHandler)
{
    ResolvedTypeMap expressionTypes;
    Visitor visitor(expression, environment, context, errorHandler);
    return expressionTypes;
}
