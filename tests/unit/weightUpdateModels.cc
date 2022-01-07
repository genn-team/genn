// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "weightUpdateModels.h"

//--------------------------------------------------------------------------
// PiecewiseSTDPCopy
//--------------------------------------------------------------------------
class PiecewiseSTDPCopy : public WeightUpdateModels::Base
{
public:
    SET_PARAM_NAMES({"tLrn", "tChng", "tDecay", "tPunish10", "tPunish01",
        "gMax", "gMid", "gSlope", "tauShift", "gSyn0"});
    SET_VARS({{"g", "scalar"}, {"gRaw", "scalar"}});

    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "scalar dt = $(sT_post) - $(t) - ($(tauShift)); \n"
        "scalar dg = 0;\n"
        "if (dt > $(lim0))  \n"
        "    dg = -($(off0)) ; \n"
        "else if (dt > 0)  \n"
        "    dg = $(slope0) * dt + ($(off1)); \n"
        "else if (dt > $(lim1))  \n"
        "    dg = $(slope1) * dt + ($(off1)); \n"
        "else dg = - ($(off2)) ; \n"
        "$(gRaw) += dg; \n"
        "$(g)=$(gMax)/2 *(tanh($(gSlope)*($(gRaw) - ($(gMid))))+1); \n");
    SET_LEARN_POST_CODE(
        "scalar dt = $(t) - ($(sT_pre)) - ($(tauShift)); \n"
        "scalar dg =0; \n"
        "if (dt > $(lim0))  \n"
        "    dg = -($(off0)) ; \n"
        "else if (dt > 0)  \n"
        "    dg = $(slope0) * dt + ($(off1)); \n"
        "else if (dt > $(lim1))  \n"
        "    dg = $(slope1) * dt + ($(off1)); \n"
        "else dg = -($(off2)) ; \n"
        "$(gRaw) += dg; \n"
        "$(g)=$(gMax)/2.0 *(tanh($(gSlope)*($(gRaw) - ($(gMid))))+1); \n");

    SET_DERIVED_PARAMS({
        {"lim0", [](const Snippet::ParamValues &pars, double){ return (1/pars["tPunish01"] + 1/pars["tChng"]) * pars["tLrn"] / (2/pars["tChng"]); }},
        {"lim1", [](const Snippet::ParamValues &pars, double){ return  -((1/pars["tPunish10"] + 1/pars["tChng"]) * pars["tLrn"] / (2/pars["tChng"])); }},
        {"slope0", [](const Snippet::ParamValues &pars, double){ return  -2*pars["gMax"]/(pars["tChng"]*pars["tLrn"]); }},
        {"slope1", [](const Snippet::ParamValues &pars, double){ return  2*pars["gMax"]/(pars["tChng"]*pars["tLrn"]); }},
        {"off0", [](const Snippet::ParamValues &pars, double){ return  pars["gMax"] / pars["tPunish01"]; }},
        {"off1", [](const Snippet::ParamValues &pars, double){ return  pars["gMax"] / pars["tChng"]; }},
        {"off2", [](const Snippet::ParamValues &pars, double){ return  pars["gMax"] / pars["tPunish10"]; }}});

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(WeightUpdateModels, CompareBuiltIn)
{
    using namespace WeightUpdateModels;

    ASSERT_EQ(StaticPulse::getInstance()->getHashDigest(), StaticPulse::getInstance()->getHashDigest());
    ASSERT_NE(StaticPulse::getInstance()->getHashDigest(), StaticPulseDendriticDelay::getInstance()->getHashDigest());
    ASSERT_NE(StaticPulse::getInstance()->getHashDigest(), StaticGraded::getInstance()->getHashDigest());
}

TEST(WeightUpdateModels, CompareCopyPasted)
{
    using namespace WeightUpdateModels;

    PiecewiseSTDPCopy pwSTDPCopy;
    ASSERT_EQ(PiecewiseSTDP::getInstance()->getHashDigest(), pwSTDPCopy.getHashDigest());
}
