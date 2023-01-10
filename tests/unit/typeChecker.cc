// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "type.h"

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
class TestErrorHandler : public ErrorHandler
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
            report(token.line, " at '" + std::string{token.lexeme} + "'", message);
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

void typeCheckStatements(std::string_view code, TypeChecker::EnvironmentExternal &typeEnvironment)
{
    // Scan
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(code, errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
 
    // Parse
    const auto statements = Parser::parseBlockItemList(tokens, errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
     
    // Typecheck
    TypeChecker::typeCheck(statements, typeEnvironment, errorHandler);
    ASSERT_FALSE(errorHandler.hasError());
}

Type::QualifiedType typeCheckExpression(std::string_view code, TypeChecker::EnvironmentExternal &typeEnvironment)
{
    // Scan
    TestErrorHandler errorHandler;
    const auto tokens = Scanner::scanSource(code, errorHandler);
    EXPECT_FALSE(errorHandler.hasError());
 
    // Parse
    const auto expression = Parser::parseExpression(tokens, errorHandler);
    EXPECT_FALSE(errorHandler.hasError());
     
    // Typecheck
    const auto type = TypeChecker::typeCheck(expression.get(), typeEnvironment, errorHandler);
    EXPECT_FALSE(errorHandler.hasError());
    return type;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(TypeChecker, ArraySubscript)
{
    // Integer array indexing
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        const auto type = typeCheckExpression("intArray[4]", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32::getInstance()->getTypeHash());
        EXPECT_FALSE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }
    
    // Float array indexing
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        typeCheckExpression("intArray[4.0f]", typeEnvironment);}, 
        TypeChecker::TypeCheckError);

    // Pointer indexing
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        typeEnvironment.define<Type::Int32Ptr>("indexArray");
        typeCheckExpression("intArray[indexArray]", typeEnvironment);}, 
        TypeChecker::TypeCheckError);
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Assignment)
{
    // Numeric assignment
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal");
        typeEnvironment.define<Type::Float>("floatVal");
        typeEnvironment.define<Type::Int32>("intValConst", true);
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

    // Pointer assignement
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        typeEnvironment.define<Type::Int32Ptr>("intArrayConst", true);
        typeCheckStatements(
            "int *x = intArray;\n"
            "const int *y = intArray;\n"
            "const int *z = intArrayConst;\n", 
            typeEnvironment);
    }

    // Pointer assignement, attempt to remove const
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray", true);
        typeCheckStatements("int *x = intArray;", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Pointer assignement without explicit cast
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        typeCheckStatements("float *x = intArray;", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // **TODO** other assignements i.e. += -= %=
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Binary)
{
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Call)
{
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Cast)
{
    // Numeric cast
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal");
        const auto type = typeCheckExpression("(float)intVal", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Float::getInstance()->getTypeHash());
        EXPECT_FALSE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Numeric cast to const
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal");
        const auto type = typeCheckExpression("(const int)intVal", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Pointer cast to value const
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        const auto type = typeCheckExpression("(const int*)intArray", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32Ptr::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Pointer cast to pointer const
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        const auto type = typeCheckExpression("(int * const)intArray", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32Ptr::getInstance()->getTypeHash());
        EXPECT_FALSE(type.constValue);
        EXPECT_TRUE(type.constPointer);
    }

    // Can't remove value const from numeric
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal", true);
        typeCheckExpression("(int)intVal", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Can't remove value const from pointer
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray", true);
        typeCheckExpression("(int*)intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);

    // Can't remove pointer const from pointer
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray", false, true);
        typeCheckExpression("(int*)intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);

   // Pointer cast can't reinterpret
   EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        typeCheckExpression("(float*)intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);
    
   // Pointer can't be cast to numeric
   EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        typeCheckExpression("(int)intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);
    
   // Numeric can't be cast to pointer
   EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal");
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
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal");
        const auto type = typeCheckExpression("intVal++", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32::getInstance()->getTypeHash());
        EXPECT_FALSE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Can increment pointer
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        const auto type = typeCheckExpression("intArray++", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32Ptr::getInstance()->getTypeHash());
        EXPECT_FALSE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Can increment pointer to const
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray", true);
        const auto type = typeCheckExpression("intArray++", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32Ptr::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

   // Can't increment const number
   EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal", true);
        typeCheckExpression("intVal++", typeEnvironment);},
        TypeChecker::TypeCheckError);
   
   // Can't increment const pointer
   EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray", false, true);
        typeCheckExpression("intArray++", typeEnvironment);},
        TypeChecker::TypeCheckError);
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Literal)
{
    // Float
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        const auto type = typeCheckExpression("1.0f", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Float::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Double
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        const auto type = typeCheckExpression("1.0", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Double::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Integer
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        const auto type = typeCheckExpression("100", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Unsigned integer
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        const auto type = typeCheckExpression("100U", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Uint32::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }
}
//--------------------------------------------------------------------------
TEST(TypeChecker, Unary)
{
    // Dereference pointer
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        const auto type = typeCheckExpression("*intArray", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32::getInstance()->getTypeHash());
        EXPECT_FALSE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Dereference pointer to const
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray", true);
        const auto type = typeCheckExpression("*intArray", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Dereference const pointer
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray", false, true);
        const auto type = typeCheckExpression("*intArray", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32::getInstance()->getTypeHash());
        EXPECT_FALSE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Dereference const pointer to const
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray", true, true);
        const auto type = typeCheckExpression("*intArray", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32::getInstance()->getTypeHash());
        EXPECT_TRUE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Dereference numeric
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal");
        typeCheckExpression("*intVal", typeEnvironment); },
        TypeChecker::TypeCheckError);

    // Address of numeric
    {
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32>("intVal");
        const auto type = typeCheckExpression("&intVal", typeEnvironment);
        EXPECT_EQ(type.type->getTypeHash(), Type::Int32Ptr::getInstance()->getTypeHash());
        EXPECT_FALSE(type.constValue);
        EXPECT_FALSE(type.constPointer);
    }

    // Address of pointer
    EXPECT_THROW({
        TypeChecker::EnvironmentExternal typeEnvironment;
        typeEnvironment.define<Type::Int32Ptr>("intArray");
        typeCheckExpression("&intArray", typeEnvironment);},
        TypeChecker::TypeCheckError);
}
