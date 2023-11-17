// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"
#include "postsynapticModels.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// ExpCurrCopy
//--------------------------------------------------------------------------
class ExpCurrCopy : public PostsynapticModels::Base
{
public:
    SET_DECAY_CODE("$(inSyn) *= $(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(init) * $(inSyn)");

    SET_PARAMS({"tau"});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const auto &pars, double dt){ return std::exp(-dt / pars.at("tau").cast<double>()); }},
        {"init", [](const auto &pars, double dt){ return (pars.at("tau").cast<double>() * (1.0 - std::exp(-dt / pars.at("tau").cast<double>()))) * (1.0 / dt); }}});
};
//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(PostsynapticModels, CompareBuiltIn)
{
    using namespace PostsynapticModels;

    ASSERT_EQ(ExpCurr::getInstance()->getHashDigest(), ExpCurr::getInstance()->getHashDigest());
    ASSERT_NE(ExpCurr::getInstance()->getHashDigest(), ExpCond::getInstance()->getHashDigest());
    ASSERT_NE(ExpCurr::getInstance()->getHashDigest(), DeltaCurr::getInstance()->getHashDigest());
}

TEST(PostsynapticModels, CompareCopyPasted)
{
    using namespace PostsynapticModels;

    ExpCurrCopy expCurrCopy;
    ASSERT_EQ(ExpCurr::getInstance()->getHashDigest(), expCurrCopy.getHashDigest());
}
