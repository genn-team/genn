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
    ASSERT_TRUE(CurrentSourceModels::DC::getInstance()->canBeMerged(CurrentSourceModels::DC::getInstance()));
    ASSERT_FALSE(CurrentSourceModels::DC::getInstance()->canBeMerged(CurrentSourceModels::GaussianNoise::getInstance()));
}

TEST(CurrentSourceModels, CompareCopyPasted)
{
    GaussianNoiseCopy gaussianCopy;
    ASSERT_TRUE(CurrentSourceModels::GaussianNoise::getInstance()->canBeMerged(&gaussianCopy));
}
