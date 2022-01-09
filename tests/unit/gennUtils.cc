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
void validatePopNameDeathTest(const std::string &name)
{
    try {
        Utils::validatePopName(name, "test");
        FAIL();
    }

    catch(const std::runtime_error &) {
    }
}
void validateVarNameDeathTest(const std::string &name)
{
    try {
        Utils::validateVarName(name, "test");
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
    Utils::validateVarName("test", "test");
    Utils::validateVarName("Test", "test");
    Utils::validateVarName("test123", "test");
    Utils::validateVarName("test_123", "test");
    Utils::validateVarName("_test_123", "test");

    Utils::validatePopName("test", "test");
    Utils::validatePopName("Test", "test");
    Utils::validatePopName("test123", "test");
    Utils::validatePopName("test_123", "test");
    Utils::validatePopName("_test_123", "test");
    Utils::validatePopName("1test", "test");

    validateVarNameDeathTest("");
    validateVarNameDeathTest("1test");
    validateVarNameDeathTest("test.test");
    validateVarNameDeathTest("test-test");
    validatePopNameDeathTest("");
    validatePopNameDeathTest("test.test");
    validatePopNameDeathTest("test-test");
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
TEST(GeNNUtils, ValidateParamValues) 
{
    const std::vector<std::string> paramNames{"a", "b", "c", "d"};
    const std::unordered_map<std::string, double> paramValsCorrect{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    const std::unordered_map<std::string, double> paramValsMisSpelled{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"D", 8.0}};
    const std::unordered_map<std::string, double> paramValsMissing{{"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    const std::unordered_map<std::string, double> paramValsExtra{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}, {"e", 8.0}};

    Utils::validateParamValues(paramNames, paramValsCorrect, "Test");

    try {
        Utils::validateParamValues(paramNames, paramValsMisSpelled, "Test");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        Utils::validateParamValues(paramNames, paramValsMissing, "Test");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        Utils::validateParamValues(paramNames, paramValsExtra, "Test");
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