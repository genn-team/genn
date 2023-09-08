// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"
#include "weightUpdateModels.h"

using namespace GeNN;

namespace
{
//--------------------------------------------------------------------------
// PiecewiseSTDPCopy
//--------------------------------------------------------------------------
class PiecewiseSTDPCopy : public WeightUpdateModels::Base
{
public:
    SET_PARAM_NAMES({"tLrn", "tChng", "tDecay", "tPunish10", "tPunish01",
                     "gMax", "gMid", "gSlope", "tauShift", "gSyn0"});
    SET_SYNAPSE_VARS({{"g", "scalar"}, {"gRaw", "scalar"}});

    SET_SIM_CODE(
        "addToPost(g);\n"
        "scalar dt = sT_post - t - tauShift; \n"
        "scalar dg = 0;\n"
        "if (dt > lim0)  \n"
        "    dg = -off0 ; \n"
        "else if (dt > 0)  \n"
        "    dg = slope0 * dt + off1; \n"
        "else if (dt > lim1)  \n"
        "    dg = slope1 * dt + ($(off1)); \n"
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
        {"lim0", [](const ParamValues &pars, double) { return (1 / pars.at("tPunish01") + 1 / pars.at("tChng")) * pars.at("tLrn") / (2 / pars.at("tChng")); }},
        {"lim1", [](const ParamValues &pars, double) { return  -((1 / pars.at("tPunish10") + 1 / pars.at("tChng")) * pars.at("tLrn") / (2 / pars.at("tChng"))); }},
        {"slope0", [](const ParamValues &pars, double) { return  -2 * pars.at("gMax") / (pars.at("tChng") * pars.at("tLrn")); }},
        {"slope1", [](const ParamValues &pars, double) { return  2 * pars.at("gMax") / (pars.at("tChng") * pars.at("tLrn")); }},
        {"off0", [](const ParamValues &pars, double) { return  pars.at("gMax") / pars.at("tPunish01"); }},
        {"off1", [](const ParamValues &pars, double) { return  pars.at("gMax") / pars.at("tChng"); }},
        {"off2", [](const ParamValues &pars, double) { return  pars.at("gMax") / pars.at("tPunish10"); }}});

};

class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditive);
    SET_PARAM_NAMES({"tauPlus", "tauMinus", "Aplus", "Aminus",
                     "Wmin", "Wmax"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauPlus")); }},
        {"tauMinusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMinus")); }}});
    SET_SYNAPSE_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    
    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "const scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) - ($(Aminus) * $(postTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "const scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) + ($(Aplus) * $(preTrace));\n"
        "    $(g) = fmax($(Wmin), fmin($(Wmax), newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("$(preTrace) += 1.0;\n");
    SET_POST_SPIKE_CODE("$(postTrace) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("$(preTrace) *= $(tauPlusDecay);\n");
    SET_POST_DYNAMICS_CODE("$(postTrace) *= $(tauMinusDecay);\n");
};
IMPLEMENT_SNIPPET(STDPAdditive);
}

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
//--------------------------------------------------------------------------
TEST(WeightUpdateModels, ValidateParamValues) 
{
    const VarValues varVals{{"g", uninitialisedVar()}};
    const VarValues preVarVals{{"preTrace", uninitialisedVar()}};
    const VarValues postVarVals{{"postTrace", uninitialisedVar()}};

    const std::unordered_map<std::string, double> paramValsCorrect{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const std::unordered_map<std::string, double> paramValsMisSpelled{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"APlus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const std::unordered_map<std::string, double> paramValsMissing{{"tauPlus", 10.0}, {"tauMinus", 10.0},{"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const std::unordered_map<std::string, double> paramValsExtra{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Bminus", 0.01},{"Wmin", 0.0}, {"Wmax", 1.0}};

    STDPAdditive::getInstance()->validate(paramValsCorrect, varVals, preVarVals, postVarVals, "Synapse group");

    try {
        STDPAdditive::getInstance()->validate(paramValsMisSpelled, varVals, preVarVals, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramValsMissing, varVals, preVarVals, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramValsExtra, varVals, preVarVals, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
//--------------------------------------------------------------------------
TEST(WeightUpdateModels, ValidateVarValues) 
{
    const VarValues preVarVals{{"preTrace", 0.0}};
    const VarValues postVarVals{{"postTrace", 0.0}};
    const std::unordered_map<std::string, double> paramVals{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsCorrect{{"g", uninitialisedVar()}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsMisSpelled{{"G", uninitialisedVar()}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsMissing{};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsExtra{{"g", uninitialisedVar()}, {"d", uninitialisedVar()}};

    STDPAdditive::getInstance()->validate(paramVals, varValsCorrect, preVarVals, postVarVals, "Synapse group");

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsMisSpelled, preVarVals, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsMissing, preVarVals, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsExtra, preVarVals, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
//--------------------------------------------------------------------------
TEST(WeightUpdateModels, ValidatePreVarValues) 
{
    const VarValues postVarVals{{"postTrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varVals{{"g", uninitialisedVar()}};
    const std::unordered_map<std::string, double> paramVals{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    
    const std::unordered_map<std::string, InitVarSnippet::Init> preVarValsCorrect{{"preTrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> preVarValsMisSpelled{{"prETrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> preVarValsMissing{};
    const std::unordered_map<std::string, InitVarSnippet::Init> preVarValsExtra{{"preTrace", 0.0}, {"postTrace", 0.0}};

    STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsCorrect, postVarVals, "Synapse group");

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsMisSpelled, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsMissing, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsExtra, postVarVals, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
//--------------------------------------------------------------------------
TEST(WeightUpdateModels, ValidatePostVarValues) 
{
    const VarValues preVarVals{{"preTrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varVals{{"g", uninitialisedVar()}};
    const std::unordered_map<std::string, double> paramVals{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    
    const std::unordered_map<std::string, InitVarSnippet::Init> postVarValsCorrect{{"postTrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> postVarValsMisSpelled{{"PostTrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> postVarValsMissing{};
    const std::unordered_map<std::string, InitVarSnippet::Init> postVarValsExtra{{"postTrace", 0.0}, {"preTrace", 0.0}};

    STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsCorrect, "Synapse group");

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsMisSpelled, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsMissing, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsExtra, "Synapse group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
