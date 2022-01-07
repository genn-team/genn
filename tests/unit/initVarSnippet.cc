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
    using namespace InitVarSnippet;

    ASSERT_EQ(Constant::getInstance()->getHashDigest(), Constant::getInstance()->getHashDigest());
    ASSERT_NE(Uniform::getInstance()->getHashDigest(), Normal::getInstance()->getHashDigest());
    ASSERT_NE(Exponential::getInstance()->getHashDigest(), Gamma::getInstance()->getHashDigest());
}

TEST(InitVarSnippet, CompareCopyPasted)
{
    using namespace InitVarSnippet;

    UniformCopy uniformCopy;
    ASSERT_EQ(Uniform::getInstance()->getHashDigest(), uniformCopy.getHashDigest());
}

TEST(InitVarSnippet, CompareVarInitParameters)
{
    Snippet::ParamValues uniformParamsA{{"min", 0.0}, {"max", 1.0}};
    Snippet::ParamValues uniformParamsB{{"min", 0.0}, {"max", 0.5}};

    const auto varInit0 = initVar<InitVarSnippet::Uniform>(uniformParamsA);
    const auto varInit1 = initVar<InitVarSnippet::Uniform>(uniformParamsA);
    const auto varInit2 = initVar<InitVarSnippet::Uniform>(uniformParamsB);

    ASSERT_EQ(varInit0.getHashDigest(), varInit1.getHashDigest());
    ASSERT_EQ(varInit0.getHashDigest(), varInit2.getHashDigest());
}
