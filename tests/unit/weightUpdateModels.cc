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
    SET_VARS({{"g", "scalar"}, {"gRaw", "scalar"}});

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
        {"lim0", [](const auto &pars, double){ return (1/pars.at("tPunish01").cast<double>() + 1 / pars.at("tChng").cast<double>()) * pars.at("tLrn").cast<double>() / (2/pars.at("tChng").cast<double>()); }},
        {"lim1", [](const auto &pars, double){ return  -((1/pars.at("tPunish10").cast<double>() + 1 / pars.at("tChng").cast<double>()) * pars.at("tLrn").cast<double>() / (2/pars.at("tChng").cast<double>())); }},
        {"slope0", [](const auto &pars, double){ return  -2*pars.at("gMax").cast<double>() /(pars.at("tChng").cast<double>()*pars.at("tLrn").cast<double>()); }},
        {"slope1", [](const auto &pars, double){ return  2*pars.at("gMax").cast<double>() / (pars.at("tChng").cast<double>() * pars.at("tLrn").cast<double>()); }},
        {"off0", [](const auto &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tPunish01").cast<double>(); }},
        {"off1", [](const auto &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tChng").cast<double>(); }},
        {"off2", [](const auto &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tPunish10").cast<double>(); }}});

};

class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditive);
    SET_PARAM_NAMES({"tauPlus", "tauMinus", "Aplus", "Aminus",
                     "Wmin", "Wmax"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const auto &pars, double dt){ return std::exp(-dt / pars.at("tauPlus").cast<double>()); }},
        {"tauMinusDecay", [](const auto &pars, double dt){ return std::exp(-dt / pars.at("tauMinus").cast<double>()); }}});
    SET_VARS({{"g", "scalar"}});
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
    const VarReferences preNeuronVarRefs{};
    const VarReferences postNeuronVarRefs{};

    const ParamValues paramValsCorrect{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const ParamValues paramValsMisSpelled{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"APlus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const ParamValues paramValsMissing{{"tauPlus", 10.0}, {"tauMinus", 10.0},{"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const ParamValues paramValsExtra{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Bminus", 0.01},{"Wmin", 0.0}, {"Wmax", 1.0}};

    STDPAdditive::getInstance()->validate(paramValsCorrect, varVals, preVarVals, postVarVals, 
                                          preNeuronVarRefs, postNeuronVarRefs);

    try {
        STDPAdditive::getInstance()->validate(paramValsMisSpelled, varVals, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramValsMissing, varVals, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramValsExtra, varVals, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
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
    const ParamValues paramVals{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const VarReferences preNeuronVarRefs{};
    const VarReferences postNeuronVarRefs{};
    
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsCorrect{{"g", uninitialisedVar()}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsMisSpelled{{"G", uninitialisedVar()}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsMissing{};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsExtra{{"g", uninitialisedVar()}, {"d", uninitialisedVar()}};

    STDPAdditive::getInstance()->validate(paramVals, varValsCorrect, preVarVals, postVarVals, 
                                          preNeuronVarRefs, postNeuronVarRefs);

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsMisSpelled, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsMissing, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsExtra, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
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
    const ParamValues paramVals{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const VarReferences preNeuronVarRefs{};
    const VarReferences postNeuronVarRefs{};
    
    const std::unordered_map<std::string, InitVarSnippet::Init> preVarValsCorrect{{"preTrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> preVarValsMisSpelled{{"prETrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> preVarValsMissing{};
    const std::unordered_map<std::string, InitVarSnippet::Init> preVarValsExtra{{"preTrace", 0.0}, {"postTrace", 0.0}};

    STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsCorrect, postVarVals, 
                                          preNeuronVarRefs, postNeuronVarRefs);

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsMisSpelled, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsMissing, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsExtra, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs);
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
    const ParamValues paramVals{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const VarReferences preNeuronVarRefs{};
    const VarReferences postNeuronVarRefs{};
    
    const std::unordered_map<std::string, InitVarSnippet::Init> postVarValsCorrect{{"postTrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> postVarValsMisSpelled{{"PostTrace", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> postVarValsMissing{};
    const std::unordered_map<std::string, InitVarSnippet::Init> postVarValsExtra{{"postTrace", 0.0}, {"preTrace", 0.0}};

    STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsCorrect, 
                                          preNeuronVarRefs, postNeuronVarRefs);

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsMisSpelled, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsMissing, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsExtra, 
                                              preNeuronVarRefs, postNeuronVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
