// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "currentSourceModels.h"

//--------------------------------------------------------------------------
// GaussianNoiseCopy
//--------------------------------------------------------------------------
class GaussianNoiseCopy : public CurrentSourceModels::Base
{
    SET_INJECTION_CODE("$(injectCurrent, $(mean) + $(gennrand_normal) * $(sd));\n");

    SET_PARAM_NAMES({"mean", "sd"} );
};

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(CurrentSourceModels, CompareBuiltIn)
{
    ASSERT_EQ(CurrentSourceModels::DC::getInstance()->getHashDigest(), CurrentSourceModels::DC::getInstance()->getHashDigest());
    ASSERT_NE(CurrentSourceModels::DC::getInstance()->getHashDigest(), CurrentSourceModels::GaussianNoise::getInstance()->getHashDigest());
}

TEST(CurrentSourceModels, CompareCopyPasted)
{
    GaussianNoiseCopy gaussianCopy;
    ASSERT_EQ(CurrentSourceModels::GaussianNoise::getInstance()->getHashDigest(), gaussianCopy.getHashDigest());
}
