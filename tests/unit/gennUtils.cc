// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "gennUtils.h"
#include "snippet.h"

using namespace GeNN;

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
//--------------------------------------------------------------------------
TEST(GeNNUtils, StringHashing)
{
    // Hash "hello" followed by empty string
    boost::uuids::detail::sha1 hash1;
    Utils::updateHash("hello", hash1);
    Utils::updateHash("", hash1);

    // Hash empty string followed by "hello"
    boost::uuids::detail::sha1 hash2;
    Utils::updateHash("", hash2);
    Utils::updateHash("hello", hash2);

    ASSERT_NE(hash1.get_digest(), hash2.get_digest());
}
//--------------------------------------------------------------------------
TEST(GeNNUtils, VectorHashing)
{
    const std::vector<float> vector1{ 1.0f, 0.0f };
    const std::vector<float> vector2;

    // Hash "hello" followed by empty string
    boost::uuids::detail::sha1 hash1;
    Utils::updateHash(vector1, hash1);
    Utils::updateHash(vector2, hash1);

    // Hash empty string followed by "hello"
    boost::uuids::detail::sha1 hash2;
    Utils::updateHash(vector2, hash2);
    Utils::updateHash(vector1, hash2);

    ASSERT_NE(hash1.get_digest(), hash2.get_digest());
}