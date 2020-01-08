// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "postsynapticModels.h"

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
    ASSERT_TRUE(PostsynapticModels::ExpCurr::getInstance()->canBeMerged(PostsynapticModels::ExpCurr::getInstance()));
    ASSERT_FALSE(PostsynapticModels::ExpCurr::getInstance()->canBeMerged(PostsynapticModels::ExpCond::getInstance()));
    ASSERT_FALSE(PostsynapticModels::ExpCurr::getInstance()->canBeMerged(PostsynapticModels::DeltaCurr::getInstance()));
}

TEST(PostsynapticModels, CompareCopyPasted)
{
    ExpCurrCopy expCurrCopy;
    ASSERT_TRUE(PostsynapticModels::ExpCurr::getInstance()->canBeMerged(&expCurrCopy));
}
