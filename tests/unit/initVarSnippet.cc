// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "initVarSnippet.h"

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
