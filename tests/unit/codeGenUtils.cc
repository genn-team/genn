// C++ standard includes
#include <limits>
#include <tuple>

// C standard includes
#include <cstdlib>

// Google test includes
#include "gtest/gtest.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/substitutions.h"

using namespace CodeGenerator;

// Test based on original issue found in https://github.com/brian-team/brian2genn/pull/60 to make sure that ensureFtype doesn't break functions it shouldn't
TEST(EnsureMathFunctionFtype, ISinF) {
    const std::string code =
        "const int _infinity_int  = 1073741823;  // maximum 32bit integer divided by 2\n"
        "if (std::isinf(t))\n"
        "{\n";

    std::string substitutedCode = ensureFtype(code, "double");
    ASSERT_EQ(code, substitutedCode);
}

// Test based on comments by Marcel in https://github.com/brian-team/brian2genn/pull/60
TEST(EnsureMathFunctionFtype, foo123) {
    const std::string code = "int foo123 = 6;";

    std::string substitutedCode = code;
    regexVarSubstitute(substitutedCode, "foo", "bar");
    ASSERT_EQ(code, substitutedCode);
}

// Test based on comments by Thomas in https://github.com/brian-team/brian2genn/pull/60
TEST(EnsureMathFunctionFtype, not2well) {
    const std::string code = "int not2well = 6;";

    std::string substitutedCode = code;
    regexVarSubstitute(substitutedCode, "well", "hell");
    ASSERT_EQ(code, substitutedCode);
}

// Test based on my own discovering this wasn't actually working
TEST(EnsureMathFunctionFtype, rint) {
    const std::string code = "$(value) = (uint8_t)rint(normal / DT);";

    std::string substitutedCode = ensureFtype(code, "float");
    ASSERT_EQ(substitutedCode, "$(value) = (uint8_t)rintf(normal / DT);");
}

//--------------------------------------------------------------------------
// SingleValueSubstitutionTest
//--------------------------------------------------------------------------
class SingleValueSubstitutionTest : public ::testing::TestWithParam<double>
{
protected:
    //--------------------------------------------------------------------------
    // Test virtuals
    //--------------------------------------------------------------------------
    virtual void SetUp()
    {
        // Substitute variable for value
        m_Code = "$(test)";

        // Substitute test parameter for value
        Substitutions subs;
        subs.addParamValueSubstitution({"test"}, { GetParam() });
        subs.apply(m_Code);
    }

    //--------------------------------------------------------------------------
    // Protected API
    //--------------------------------------------------------------------------
    const std::string &GetCode() const { return m_Code; }

private:
    //--------------------------------------------------------------------------
    // Private API
    //--------------------------------------------------------------------------
    std::string m_Code;
};

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST_P(SingleValueSubstitutionTest, CorrectGeneratedValue)
{
    // Convert results back to double and check they match
    double result = std::atof(GetCode().c_str());
    ASSERT_DOUBLE_EQ(result, GetParam());
}

//--------------------------------------------------------------------------
// Instatiations
//--------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(DoubleValues,
                        SingleValueSubstitutionTest,
                        ::testing::Values(std::numeric_limits<double>::min(),
                                          std::numeric_limits<double>::max(),
                                          1.0,
                                          -1.0),);
