// C++ standard includes
#include <limits>
#include <tuple>

// C standard includes
#include <cstdlib>

// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "codeGenUtils.h"

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
        std::vector<std::string> names = {"test"};
        std::vector<double> values = { GetParam() };
        value_substitutions(m_Code, names, values);

        // For safety, value_substitutions adds brackets around substituted values - trim these out
        m_Code = m_Code.substr(1, m_Code.size() - 2);
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
                                          -1.0));