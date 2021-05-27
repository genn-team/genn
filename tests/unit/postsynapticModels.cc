// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "postsynapticModels.h"

// Unit test includes
#include "hashUtils.h"

//--------------------------------------------------------------------------
// ExpCurrCopy
//--------------------------------------------------------------------------
class ExpCurrCopy : public PostsynapticModels::Base
{
public:
    SET_DECAY_CODE("$(inSyn) *= $(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(init) * $(inSyn)");

    SET_PARAM_NAMES({"tau"});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"init", [](const std::vector<double> &pars, double dt){ return (pars[0] * (1.0 - std::exp(-dt / pars[0]))) * (1.0 / dt); }}});
};
//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(PostsynapticModels, CompareBuiltIn)
{
    using namespace PostsynapticModels;

    ASSERT_TRUE(ExpCurr::getInstance()->canBeMerged(ExpCurr::getInstance()));
    ASSERT_FALSE(ExpCurr::getInstance()->canBeMerged(ExpCond::getInstance()));
    ASSERT_FALSE(ExpCurr::getInstance()->canBeMerged(DeltaCurr::getInstance()));

    ASSERT_MODEL_HASH_EQ(ExpCurr::getInstance(), ExpCurr::getInstance());
    ASSERT_MODEL_HASH_NE(ExpCurr::getInstance(), ExpCond::getInstance());
    ASSERT_MODEL_HASH_NE(ExpCurr::getInstance(), DeltaCurr::getInstance());
}

TEST(PostsynapticModels, CompareCopyPasted)
{
    using namespace PostsynapticModels;

    ExpCurrCopy expCurrCopy;
    ASSERT_TRUE(ExpCurr::getInstance()->canBeMerged(&expCurrCopy));

    ASSERT_MODEL_HASH_EQ(ExpCurr::getInstance(), &expCurrCopy);
}
