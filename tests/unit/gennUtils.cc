// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "gennUtils.h"
#include "snippet.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void validateVarPopNameDeathTest(const std::string &name)
{
    try {
        Utils::validateVarPopName(name, "test");
        FAIL();
    }

    catch(const std::runtime_error &) {
    }
}
}

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(GeNNUtils, ValidateVarPopName)
{
    Utils::validateVarPopName("test", "test");
    Utils::validateVarPopName("Test", "test");
    Utils::validateVarPopName("test123", "test");
    Utils::validateVarPopName("test_123", "test");
    Utils::validateVarPopName("_test_123", "test");

    validateVarPopNameDeathTest("");
    validateVarPopNameDeathTest("1test");
    validateVarPopNameDeathTest("test.test");
    validateVarPopNameDeathTest("test-test");
}
//--------------------------------------------------------------------------
TEST(GeNNUtils, ValidateParamNames)
{
    Utils::validateParamNames({"test", "Test", "test123"});

    try {
        Utils::validateParamNames({"test", "test.test"});
        FAIL();
    }

    catch(const std::runtime_error &) {
    }
}
//--------------------------------------------------------------------------
TEST(GeNNUtils, ValidateVecNames)
{
    const Snippet::Base::ParamValVec good{{"test", "scalar", 1.0}, {"Test", "scalar", 0.0}};
    Utils::validateVecNames(good, "test");

    try {
        const Snippet::Base::ParamValVec bad{{"test", "scalar", 1.0}, {"test.test", "scalar", 0.0}};
        Utils::validateVecNames(bad, "test");
        FAIL();
    }

    catch(const std::runtime_error &) {
    }
}