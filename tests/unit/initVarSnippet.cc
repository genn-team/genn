// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"

// Unit test includes
#include "hashUtils.h"

//--------------------------------------------------------------------------
// UniformCopy
//--------------------------------------------------------------------------
class UniformCopy : public InitVarSnippet::Base
{
public:
    SET_CODE(
        "const scalar scale = $(max) - $(min);\n"
        "$(value) = $(min) + ($(gennrand_uniform) * scale);");

    SET_PARAM_NAMES({"min", "max"});
};

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(InitVarSnippet, CompareBuiltIn)
{
    using namespace InitVarSnippet;

    ASSERT_TRUE(Constant::getInstance()->canBeMerged(Constant::getInstance()));
    ASSERT_FALSE(Uniform::getInstance()->canBeMerged(Normal::getInstance()));
    ASSERT_FALSE(Exponential::getInstance()->canBeMerged(Gamma::getInstance()));

    ASSERT_HASH_EQ(Constant::getInstance(), Constant::getInstance(), updateHash);
    ASSERT_HASH_NE(Uniform::getInstance(), Normal::getInstance(), updateHash);
    ASSERT_HASH_NE(Exponential::getInstance(), Gamma::getInstance(), updateHash);
}

TEST(InitVarSnippet, CompareCopyPasted)
{
    using namespace InitVarSnippet;

    UniformCopy uniformCopy;
    ASSERT_TRUE(Uniform::getInstance()->canBeMerged(&uniformCopy));
    ASSERT_HASH_EQ(Uniform::getInstance(), &uniformCopy, updateHash);
}

TEST(InitVarSnippet, CompareVarInitParameters)
{
    InitVarSnippet::Uniform::ParamValues uniformParamsA(0.0, 1.0);
    InitVarSnippet::Uniform::ParamValues uniformParamsB(0.0, 0.5);

    const auto varInit0 = initVar<InitVarSnippet::Uniform>(uniformParamsA);
    const auto varInit1 = initVar<InitVarSnippet::Uniform>(uniformParamsA);
    const auto varInit2 = initVar<InitVarSnippet::Uniform>(uniformParamsB);

    ASSERT_TRUE(varInit0.canBeMerged(varInit1));
    ASSERT_TRUE(varInit0.canBeMerged(varInit2));
}
