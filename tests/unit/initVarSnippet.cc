// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"

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
    ASSERT_TRUE(InitVarSnippet::Constant::getInstance()->canBeMerged(InitVarSnippet::Constant::getInstance()));
    ASSERT_FALSE(InitVarSnippet::Uniform::getInstance()->canBeMerged(InitVarSnippet::Normal::getInstance()));
    ASSERT_FALSE(InitVarSnippet::Exponential::getInstance()->canBeMerged(InitVarSnippet::Gamma::getInstance()));
}

TEST(InitVarSnippet, CompareCopyPasted)
{
    UniformCopy uniformCopy;
    ASSERT_TRUE(InitVarSnippet::Uniform::getInstance()->canBeMerged(&uniformCopy));
}

TEST(InitVarSnippet, CompareVarInitParameters)
{
    InitVarSnippet::Uniform::ParamValues uniformParamsA(0.0, 1.0);
    InitVarSnippet::Uniform::ParamValues uniformParamsB(0.0, 0.5);

    const auto varInit0 = initVar<InitVarSnippet::Uniform>(uniformParamsA);
    const auto varInit1 = initVar<InitVarSnippet::Uniform>(uniformParamsA);
    const auto varInit2 = initVar<InitVarSnippet::Uniform>(uniformParamsB);

    ASSERT_TRUE(varInit0.canBeMerged(varInit1));
    ASSERT_FALSE(varInit0.canBeMerged(varInit2));
}
