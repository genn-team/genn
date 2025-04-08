// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "type.h"

// GeNN code generator includes
#include "code_generator/standardLibrary.h"

// GeNN transpiler includes
#include "transpiler/errorHandler.h"
#include "transpiler/parser.h"
#include "transpiler/scanner.h"
#include "transpiler/typeChecker.h"

using namespace GeNN;
using namespace GeNN::Transpiler;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
class TestErrorHandler : public ErrorHandlerBase
{
public:
    TestErrorHandler() : m_Error(false)
    {}

    bool hasError() const { return m_Error; }

    virtual void error(size_t line, std::string_view message) override
    {
        report(line, "", message);
    }

    virtual void error(const Token &token, std::string_view message) override
    {
        if(token.type == Token::Type::END_OF_FILE) {
            report(token.line, " at end", message);
        }
        else {
            report(token.line, " at '" + token.lexeme + "'", message);
        }
    }

private:
    void report(size_t line, std::string_view where, std::string_view message)
    {
        std::cerr << "[line " << line << "] Error" << where << ": " << message << std::endl;
        m_Error = true;
    }

    bool m_Error;
};

class TestEnvironment : public TypeChecker::EnvironmentBase
{
public:
    //---------------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------------
    void define(const Type::ResolvedType &type, const std::string &name)
    {
        if(!m_Types.try_emplace(name, type).second) {
            throw std::runtime_error("Redeclaration of '" + std::string{name} + "'");
        }
    }

    //---------------------------------------------------------------------------
    // EnvironmentBase virtuals
    //---------------------------------------------------------------------------
    virtual void define(const Token &name, const Type::ResolvedType&, ErrorHandlerBase &errorHandler) final
    {
        errorHandler.error(name, "Cannot declare variable in external environment");
        throw TypeChecker::TypeCheckError();
    }

    virtual std::vector<Type::ResolvedType> getTypes(const Token &name, ErrorHandlerBase &errorHandler) final
    {
        auto type = m_Types.find(std::string{name.lexeme});
        if(type == m_Types.end()) {
            errorHandler.error(name, "Undefined variable");
            throw TypeChecker::TypeCheckError();
        }
        else {
            return {type->second};
        }
    }

private:
    //---------------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------------
    std::unordered_map<std::string, Type::ResolvedType> m_Types;
};

class TestLibraryEnvironment : public TypeChecker::EnvironmentBase
{
public:
    explicit TestLibraryEnvironment(const CodeGenerator::EnvironmentLibrary::Library &library)
    :   m_Library(library)
    {}
    //---------------------------------------------------------------------------
    // EnvironmentBase virtuals
    //---------------------------------------------------------------------------
    virtual void define(const Token&, const Type::ResolvedType&, ErrorHandlerBase&) final
    {
        throw TypeChecker::TypeCheckError();
    }

    virtual std::vector<Type::ResolvedType> getTypes(const Token &name, ErrorHandlerBase&) final
    {
        const auto [typeBegin, typeEnd] = m_Library.get().equal_range(name.lexeme);
        if (typeBegin == typeEnd) {
             throw TypeChecker::TypeCheckError();
        }
        else {
            std::vector<Type::ResolvedType> types;
            types.reserve(std::distance(typeBegin, typeEnd));
            std::transform(typeBegin, typeEnd, std::back_inserter(types),
                           [](const auto &t) { return t.second.first; });
            return types;
        }
    }
private:
    std::reference_wrapper<const CodeGenerator::EnvironmentLibrary::Library> m_Library;
};

void typeCheckStatements(std::string_view code, TypeChecker::EnvironmentBase &typeEnvironment, const Type::TypeContext &typeContext = {})
{
    // Scan
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(code, errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
 
    // Parse
    const auto statements = Parser::parseBlockItemList(tokens, typeContext, errorHandler);
    ASSERT_FALSE(errorHandler.hasError());

    // Typecheck
    TypeChecker::EnvironmentInternal typeEnvironmentInternal(typeEnvironment);
    TypeChecker::typeCheck(statements, typeEnvironmentInternal, typeContext, errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
}

Type::ResolvedType typeCheckExpression(std::string_view code, TypeChecker::EnvironmentBase &typeEnvironment, const Type::TypeContext &typeContext = {})
{
    // Scan
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(code, errorHandler);
    EXPECT_FALSE(errorHandler.hasError());
 
    // Parse
    const auto expression = Parser::parseExpression(tokens, typeContext, errorHandler);
    EXPECT_FALSE(errorHandler.hasError());
    
    // Typecheck
    TypeChecker::EnvironmentInternal typeEnvironmentInternal(typeEnvironment);
    const auto resolvedTypes = TypeChecker::typeCheck(expression.get(), typeEnvironmentInternal, typeContext, errorHandler);
    EXPECT_FALSE(errorHandler.hasError());
    return resolvedTypes.at(expression.get());
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(TypeChecker, ArraySubscript)
{
    // Integer array indexing
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        const auto type = typeCheckExpression("intArray[4]", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
        EXPECT_FALSE(type.isConst);
    }

    // Pointer to pointer, double indexing
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer().createPointer(), "intPtrArray");
        const auto type = typeCheckExpression("intPtrArray[4][4]", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
        EXPECT_FALSE(type.isConst);
    }


    // Array-subscript overload indexing
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::ResolvedType::createFunction(Type::Int32, {Type::Uint32},
                                                                  Type::FunctionFlags::ARRAY_SUBSCRIPT_OVERRIDE),
                               "overrideIntArray");
        const auto type = typeCheckExpression("overrideIntArray[4]", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
        EXPECT_FALSE(type.isConst);
    }

    // Array-subscript overloading indexing followed by standard indexing
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::ResolvedType::createFunction(Type::Int32.createPointer(), {Type::Uint32},
                                                                  Type::FunctionFlags::ARRAY_SUBSCRIPT_OVERRIDE),
                               "overrideIntArray");
        const auto type = typeCheckExpression("overrideIntArray[4][4]", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
        EXPECT_FALSE(type.isConst);
    }

    // Call standard function with []
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::ResolvedType::createFunction(Type::Int32, {Type::Uint32}),
                               "overrideIntArray");
        typeCheckExpression("overrideIntArray[4]", typeEnvironment);}, 
        TypeChecker::TypeCheckError);

    // Call array-subscript operator overload with ()
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::ResolvedType::createFunction(Type::Int32, {Type::Uint32},
                                                                  Type::FunctionFlags::ARRAY_SUBSCRIPT_OVERRIDE),
                               "overrideIntArray");
        typeCheckExpression("overrideIntArray(4)", typeEnvironment);}, 
        TypeChecker::TypeCheckError);

    // Float array indexing
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeCheckExpression("intArray[4.0f]", typeEnvironment);}, 
        TypeChecker::TypeCheckError);

    // Pointer indexing
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeEnvironment.define(Type::Int32.createPointer(), "indexArray");
        typeCheckExpression("intArray[indexArray]", typeEnvironment);}, 
        TypeChecker::TypeCheckError);

    // Write-only indexing
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeEnvironment.define(Type::Int32.addWriteOnly(), "index");
        typeCheckExpression("intArray[index]", typeEnvironment);}, 
        TypeChecker::TypeCheckError);
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Assignment)
{
    // Dereference assignment
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeCheckExpression("*intArray = 7", typeEnvironment);
    }

    // Array subscript assignment
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeCheckExpression("intArray[5] = 7", typeEnvironment);
    }

    // Write-only assignment
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addWriteOnly(), "test");
        typeCheckExpression("test = 7", typeEnvironment);
    }

    // Write-only += assignement
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addWriteOnly(), "test");
        typeCheckExpression("test += 7", typeEnvironment);}, 
        TypeChecker::TypeCheckError);
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Binary)
{
    // Pointer difference
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray1");
        typeEnvironment.define(Type::Int32.createPointer(), "intArray2");
        const auto type = typeCheckExpression("intArray1 - intArray2", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
    }

    // Integer + integer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "a");
        typeEnvironment.define(Type::Int32, "b");
        const auto type = typeCheckExpression("a + b", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
    }

    // Small integer + small integer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int8, "a");
        typeEnvironment.define(Type::Int16, "b");
        const auto type = typeCheckExpression("a + b", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
    }

    // Integer + floating point
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Float, "a");
        typeEnvironment.define(Type::Int32, "b");
        const auto type = typeCheckExpression("a + b", typeEnvironment);
        EXPECT_EQ(type, Type::Float);
    }

    // Integer + fixed point
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::S8_7, "a");
        typeEnvironment.define(Type::Int32, "b");
        const auto type = typeCheckExpression("a + b", typeEnvironment);
        EXPECT_EQ(type, Type::S8_7);
    }

    // Floating point + fixed point
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::S8_7, "a");
        typeEnvironment.define(Type::Float, "b");
        const auto type = typeCheckExpression("a + b", typeEnvironment);
        EXPECT_EQ(type, Type::Float);
    }

    // Fixed point + fixed point
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::S8_7, "a");
        typeEnvironment.define(Type::S0_15, "b");
        const auto type = typeCheckExpression("a + b", typeEnvironment);
        EXPECT_EQ(type, Type::S8_7);
    }

    // Fixed point + saturating fixed point
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::S8_7, "a");
        typeEnvironment.define(Type::S0_15Sat, "b");
        const auto type = typeCheckExpression("a + b", typeEnvironment);
        EXPECT_EQ(type, Type::S8_7Sat);
    }

    // Pointer + integer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeEnvironment.define(Type::Int32, "offset");
        const auto type = typeCheckExpression("intArray + offset", typeEnvironment);
        EXPECT_EQ(*type.getPointer().valueType, Type::Int32);       
    }

    // **TODO** constness and 

    // Pointer + non-integer
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeEnvironment.define(Type::Float, "offset");
        typeCheckExpression("intArray + offset", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Pointer + pointer
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray1");
        typeEnvironment.define(Type::Int32.createPointer(), "intArray2");
        typeCheckExpression("intArray1 + intArray2", typeEnvironment);},
        TypeChecker::TypeCheckError);


    // Pointer - integer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeEnvironment.define(Type::Int32, "offset");
        const auto type = typeCheckExpression("intArray - offset", typeEnvironment);
        EXPECT_EQ(*type.getPointer().valueType, Type::Int32);
    }

    // Integer + pointer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeEnvironment.define(Type::Int32, "offset");
        const auto type = typeCheckExpression("offset + intArray", typeEnvironment);
        EXPECT_EQ(*type.getPointer().valueType, Type::Int32);       
    }

    // Integer + write only integer
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addWriteOnly(), "int1");
        typeEnvironment.define(Type::Int32, "int2");
        typeCheckExpression("int1 + int2", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Write only integer + integer
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "int1");
        typeEnvironment.define(Type::Int32.addWriteOnly(), "int2");
        typeCheckExpression("int1 + int2", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Integer << integer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "a");
        typeEnvironment.define(Type::Int32, "b");
        const auto type = typeCheckExpression("a << b", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
    }

    // Fixed point << integer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::S8_7, "a");
        typeEnvironment.define(Type::Int32, "b");
        const auto type = typeCheckExpression("a << b", typeEnvironment);
        EXPECT_EQ(type, Type::S8_7);
    }

    // Float << integer
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Float, "a");
        typeEnvironment.define(Type::Int32, "b");
        typeCheckExpression("a << b", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Modulus float
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Float, "a");
        typeEnvironment.define(Type::Float, "b");
        typeCheckExpression("a % b", typeEnvironment);},
        TypeChecker::TypeCheckError);
    
    
    // Bitwise fixed
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::S0_15, "a");
        typeEnvironment.define(Type::S0_15, "b");
        typeCheckExpression("a ^ b", typeEnvironment);},
        TypeChecker::TypeCheckError);

    /*integer only (opType == Token::Type::PERCENT || opType == Token::Type::SHIFT_LEFT
                    || opType == Token::Type::SHIFT_RIGHT || opType == Token::Type::CARET
                    || opType == Token::Type::AMPERSAND || opType == Token::Type::PIPE)*/

}
//--------------------------------------------------------------------------
TEST(TypeChecker, Call)
{
    // Too few arguments
    TestLibraryEnvironment stdLibraryEnv(CodeGenerator::StandardLibrary::getMathsFunctions());
    EXPECT_THROW({
        typeCheckExpression("sin()", stdLibraryEnv);}, 
        TypeChecker::TypeCheckError);

    // Too many arguments
    EXPECT_THROW({
        typeCheckExpression("sin(1.0f, 2.0f)", stdLibraryEnv);}, 
        TypeChecker::TypeCheckError);

    // Floating point transcendental function
    {
        const auto type = typeCheckExpression("sin(1.0f)", stdLibraryEnv);
        EXPECT_EQ(type, Type::Float);
    }

    // Double transcendental function
    {
        const auto type = typeCheckExpression("sin(1.0d)", stdLibraryEnv);
        EXPECT_EQ(type, Type::Double);
    }

    // Float scalar transcendental function
    {
        const Type::TypeContext typeContext{{"scalar", Type::Float}};
        const auto type = typeCheckExpression("sin(1.0)", stdLibraryEnv, typeContext);
        EXPECT_EQ(type, Type::Float);
    }

    // Double scalar transcendental function
    {
        const Type::TypeContext typeContext{{"scalar", Type::Double}};
        const auto type = typeCheckExpression("sin(1.0)", stdLibraryEnv, typeContext);
        EXPECT_EQ(type, Type::Double);
    }

    // Nested transcendental function
    {
        const auto type = typeCheckExpression("sin(fmax(0.0f, 1.0f))", stdLibraryEnv);
        EXPECT_EQ(type, Type::Float);
    }

    // Variadic with too few arguments
    EXPECT_THROW({
        typeCheckExpression("printf()", stdLibraryEnv);},
        TypeChecker::TypeCheckError);

    // Variadic function with no extra arguments
    {
        const auto type = typeCheckExpression("printf(\"hello world\")", stdLibraryEnv);
        EXPECT_EQ(type, Type::Int32);
    }

    // Variadic function with extra arguments
    {
        const auto type = typeCheckExpression("printf(\"hello world %d, %f\", 12, cos(5.0f))", stdLibraryEnv);
        EXPECT_EQ(type, Type::Int32);
    }

     // Function call with write-only argument
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::ResolvedType::createFunction(Type::Int32, {Type::Int32}), "func");
        typeEnvironment.define(Type::Int32.addWriteOnly(), "int2");
        typeCheckExpression("func(int2)", typeEnvironment);},
        TypeChecker::TypeCheckError);
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Cast)
{
    // Numeric cast
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "intVal");
        const auto type = typeCheckExpression("(float)intVal", typeEnvironment);
        EXPECT_EQ(type, Type::Float);
    }

    // Numeric cast to const
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "intVal");
        const auto type = typeCheckExpression("(const int)intVal", typeEnvironment);
        EXPECT_EQ(type, Type::Int32.addConst());
    }

    // Pointer cast to value const
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        const auto type = typeCheckExpression("(const int*)intArray", typeEnvironment);
        EXPECT_FALSE(type.isConst);
        EXPECT_EQ(*type.getPointer().valueType, Type::Int32.addConst());
    }

    // Pointer cast to pointer const
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        const auto type = typeCheckExpression("(int * const)intArray", typeEnvironment);
        EXPECT_TRUE(type.isConst);
        EXPECT_EQ(*type.getPointer().valueType, Type::Int32);    
    }

    // Can't remove value const from pointer
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addConst().createPointer(), "intArray");
        typeCheckExpression("(int*)intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Can't remove pointer const from pointer
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(true), "intArray");
        typeCheckExpression("(int*)intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);

   // Pointer cast can't reinterpret
   EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeCheckExpression("(float*)intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);
    
   // Pointer can't be cast to numeric
   EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeCheckExpression("(int)intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);
    
   // Numeric can't be cast to pointer
   EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "intVal");
        typeCheckExpression("(int*)intVal", typeEnvironment);},
        TypeChecker::TypeCheckError);
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Conditional)
{
}
//--------------------------------------------------------------------------
TEST(TypeChecker, IncDec)
{
    // Can increment numeric
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "intVal");
        const auto type = typeCheckExpression("intVal++", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
        EXPECT_FALSE(type.isConst);
    }

    // Can increment const int* pointer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        const auto type = typeCheckExpression("intArray++", typeEnvironment);
        EXPECT_EQ(type, Type::Int32.createPointer());
        EXPECT_FALSE(type.isConst);
    }

    // Can increment pointer to const
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addConst().createPointer(), "intArray");
        const auto type = typeCheckExpression("intArray++", typeEnvironment);
        EXPECT_FALSE(type.isConst);
        EXPECT_EQ(*type.getPointer().valueType, Type::Int32.addConst());
    }

   // Can't increment const number
   EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addConst(), "intVal");
        typeCheckExpression("intVal++", typeEnvironment);},
        TypeChecker::TypeCheckError);
   
   // Can't increment int * const pointer
   EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(true), "intArray");
        typeCheckExpression("intArray++", typeEnvironment);},
        TypeChecker::TypeCheckError);
    
   // Can't increment write-only number
   EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addWriteOnly(), "intVal");
        typeCheckExpression("intVal++", typeEnvironment);},
        TypeChecker::TypeCheckError);
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Literal)
{
    // Float
    {
        TestEnvironment typeEnvironment;
        const auto type = typeCheckExpression("1.0f", typeEnvironment);
        EXPECT_EQ(type, Type::Float);
    }

    // Fixed point "accum"
    {
        TestEnvironment typeEnvironment;
        const auto type = typeCheckExpression("1.0hk", typeEnvironment);
        EXPECT_EQ(type, Type::S8_7);
    }

    // Fixed point "fract"
    {
        TestEnvironment typeEnvironment;
        const auto type = typeCheckExpression("1.0hr", typeEnvironment);
        EXPECT_EQ(type, Type::S0_15);
    }

    // Scalar with single-precision
    {
        TestEnvironment typeEnvironment;
        const Type::TypeContext typeContext{{"scalar", Type::Float}};
        const auto type = typeCheckExpression("1.0", typeEnvironment, typeContext);
        EXPECT_EQ(type, Type::Float);
    }

    // Scalar with double-precision
    {
        TestEnvironment typeEnvironment;
        const Type::TypeContext typeContext{{"scalar", Type::Double}};
        const auto type = typeCheckExpression("1.0", typeEnvironment, typeContext);
        EXPECT_EQ(type, Type::Double);
    }

    // Double
    {
        TestEnvironment typeEnvironment;
        const auto type = typeCheckExpression("1.0d", typeEnvironment);
        EXPECT_EQ(type, Type::Double);
    }

    // Integer
    {
        TestEnvironment typeEnvironment;
        const auto type = typeCheckExpression("100", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
    }

    // Unsigned integer
    {
        TestEnvironment typeEnvironment;
        const auto type = typeCheckExpression("100U", typeEnvironment);
        EXPECT_EQ(type, Type::Uint32);
    }

    // String
    {
        TestEnvironment typeEnvironment;
        const auto type = typeCheckExpression("\"hello world\"", typeEnvironment);
        EXPECT_EQ(type, Type::Int8.createPointer(true));
    }
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Unary)
{
    // Dereference pointer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        const auto type = typeCheckExpression("*intArray", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
    }

    // Dereference pointer to const
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addConst().createPointer(), "intArray");
        const auto type = typeCheckExpression("*intArray", typeEnvironment);
        EXPECT_EQ(type, Type::Int32.addConst());
    }

    // Dereference const pointer
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(true), "intArray");
        const auto type = typeCheckExpression("*intArray", typeEnvironment);
        EXPECT_EQ(type, Type::Int32);
    }

    // Dereference const pointer to const
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addConst().createPointer(true), "intArray");
        const auto type = typeCheckExpression("*intArray", typeEnvironment);
        EXPECT_EQ(type, Type::Int32.addConst());
    }

    // Dereference numeric
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "intVal");
        typeCheckExpression("*intVal", typeEnvironment); },
        TypeChecker::TypeCheckError);
}
//--------------------------------------------------------------------------
TEST(TypeChecker, VarDeclaration)
{
    // Numeric var declaration
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32, "intVal");
        typeEnvironment.define(Type::Float, "floatVal");
        typeEnvironment.define(Type::Int32.addConst(), "intValConst");
        typeCheckStatements(
            "int w = intVal;\n"
            "float x = floatVal;\n"
            "int y = floatVal;\n"
            "float z = intVal;\n"
            "int wc = intValConst;\n"
            "const int cw = intVal;\n"
            "const int cwc = intValConst;\n",
            typeEnvironment);
    }

    // Pointer var declaration
    {
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeEnvironment.define(Type::Int32.addConst().createPointer(), "intArrayConst");
        typeCheckStatements(
            "int *x = intArray;\n"
            "const int *y = intArray;\n"
            "const int *z = intArrayConst;\n", 
            typeEnvironment);
    }

    // Pointer var declaration, attempt to remove const
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.addConst().createPointer(), "intArray");
        typeCheckStatements("int *x = intArray;", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Pointer var declaration without explicit cast
    EXPECT_THROW({
        TestEnvironment typeEnvironment;
        typeEnvironment.define(Type::Int32.createPointer(), "intArray");
        typeCheckStatements("float *x = intArray;", typeEnvironment);},
        TypeChecker::TypeCheckError);
}