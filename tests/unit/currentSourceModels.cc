// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// GaussianNoiseCopy
//--------------------------------------------------------------------------
class GaussianNoiseCopy : public CurrentSourceModels::Base
{
    SET_INJECTION_CODE("injectCurrent(mean + (gennrand_normal() * sd));\n");

    SET_PARAMS({"mean", "sd"} );
};

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(CurrentSourceModels, CompareBuiltIn)
{
    ASSERT_EQ(CurrentSourceModels::DC::getInstance()->getHashDigest(), CurrentSourceModels::DC::getInstance()->getHashDigest());
    ASSERT_NE(CurrentSourceModels::DC::getInstance()->getHashDigest(), CurrentSourceModels::GaussianNoise::getInstance()->getHashDigest());
}
//--------------------------------------------------------------------------
TEST(CurrentSourceModels, CompareCopyPasted)
{
    GaussianNoiseCopy gaussianCopy;
    ASSERT_EQ(CurrentSourceModels::GaussianNoise::getInstance()->getHashDigest(), gaussianCopy.getHashDigest());
}
//--------------------------------------------------------------------------
TEST(CurrentSourceModels, ValidateParamValues) 
{
    const ParamValues paramValsCorrect{{"mean", 0.0}, {"sd", 1.0}};
    const ParamValues paramValsMisSpelled{{"means", 0.0}, {"sd", 1.0}};
    const ParamValues paramValsMissing{{"mean", 0.0}};
    const ParamValues paramValsExtra{{"mean", 0.0}, {"sd", 1.0}, {"var", 1.0}};

    CurrentSourceModels::GaussianNoise::getInstance()->validate(paramValsCorrect, {}, {}, {}, "Current source");

    try {
        CurrentSourceModels::GaussianNoise::getInstance()->validate(paramValsMisSpelled, {}, {}, {}, "Current source");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        CurrentSourceModels::GaussianNoise::getInstance()->validate(paramValsMissing, {}, {}, {}, "Current source");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        CurrentSourceModels::GaussianNoise::getInstance()->validate(paramValsExtra, {}, {}, {}, "Current source");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
//--------------------------------------------------------------------------
TEST(CurrentSourceModels, ValidateVarValues) 
{
    const ParamValues paramVals{{"weight", 1.0}, {"tauSyn", 5.0}, {"rate", 10.0}};
    
    const VarValues varValsCorrect{{"current", 0.0}};
    const VarValues varValsMisSpelled{{"currents", 0.0}};
    const VarValues varValsMissing{};
    const VarValues varValsExtra{{"current", 0.0}, {"power", 1.0}};

    CurrentSourceModels::PoissonExp::getInstance()->validate(paramVals, varValsCorrect, {}, {}, "Current source");

    try {
        CurrentSourceModels::PoissonExp::getInstance()->validate(paramVals, varValsMisSpelled, {}, {}, "Current source");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        CurrentSourceModels::PoissonExp::getInstance()->validate(paramVals, varValsMissing, {}, {}, "Current source");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        CurrentSourceModels::PoissonExp::getInstance()->validate(paramVals, varValsExtra, {}, {}, "Current source");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
