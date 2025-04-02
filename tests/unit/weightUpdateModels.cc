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
    SET_PARAMS({"tLrn", "tChng", "tDecay", "tPunish10", "tPunish01",
                     "gMax", "gMid", "gSlope", "tauShift", "gSyn0"});
    SET_VARS({{"g", "scalar"}, {"gRaw", "scalar"}});

    SET_PRE_SPIKE_SYN_CODE(
        "addToPost(g);\n"
        "scalar dt = st_post - t - tauShift; \n"
        "scalar dg = 0;\n"
        "if (dt > lim0)  \n"
        "    dg = -off0 ; \n"
        "else if (dt > 0)  \n"
        "    dg = slope0 * dt + off1; \n"
        "else if (dt > lim1)  \n"
        "    dg = slope1 * dt + (off1); \n"
        "else dg = - (off2) ; \n"
        "gRaw += dg; \n"
        "g=gMax/2 *(tanh(gSlope*(gRaw - (gMid)))+1); \n");
    SET_POST_SPIKE_SYN_CODE(
        "scalar dt = t - st_pre - (tauShift); \n"
        "scalar dg =0; \n"
        "if (dt > lim0)  \n"
        "    dg = -(off0) ; \n"
        "else if (dt > 0)  \n"
        "    dg = slope0 * dt + (off1); \n"
        "else if (dt > lim1)  \n"
        "    dg = slope1 * dt + (off1); \n"
        "else dg = -(off2) ; \n"
        "gRaw += dg; \n"
        "g=gMax/2.0 *(tanh(gSlope*(gRaw - (gMid)))+1); \n");

    SET_DERIVED_PARAMS({
        {"lim0", [](const ParamValues &pars, double){ return (1/pars.at("tPunish01").cast<double>() + 1 / pars.at("tChng").cast<double>()) * pars.at("tLrn").cast<double>() / (2/pars.at("tChng").cast<double>()); }},
        {"lim1", [](const ParamValues &pars, double){ return  -((1/pars.at("tPunish10").cast<double>() + 1 / pars.at("tChng").cast<double>()) * pars.at("tLrn").cast<double>() / (2/pars.at("tChng").cast<double>())); }},
        {"slope0", [](const ParamValues &pars, double){ return  -2*pars.at("gMax").cast<double>() /(pars.at("tChng").cast<double>()*pars.at("tLrn").cast<double>()); }},
        {"slope1", [](const ParamValues &pars, double){ return  2*pars.at("gMax").cast<double>() / (pars.at("tChng").cast<double>() * pars.at("tLrn").cast<double>()); }},
        {"off0", [](const ParamValues &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tPunish01").cast<double>(); }},
        {"off1", [](const ParamValues &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tChng").cast<double>(); }},
        {"off2", [](const ParamValues &pars, double){ return  pars.at("gMax").cast<double>() / pars.at("tPunish10").cast<double>(); }}});

};

class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(STDPAdditive);
    SET_PARAMS({"tauPlus", "tauMinus", "Aplus", "Aminus",
                "Wmin", "Wmax"});
    SET_DERIVED_PARAMS({
        {"tauPlusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauPlus").cast<double>()); }},
        {"tauMinusDecay", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMinus").cast<double>()); }}});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});
    
    SET_PRE_SPIKE_SYN_CODE(
        "addToPost(g);\n"
        "const scalar dt = t - st_post; \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = g - (Aminus * postTrace);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_POST_SPIKE_SYN_CODE(
        "const scalar dt = t - st_pre;\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = g + (Aplus * preTrace);\n"
        "    g = fmax(Wmin, fmin(Wmax, newWeight));\n"
        "}\n");
    SET_PRE_SPIKE_CODE("preTrace += 1.0;\n");
    SET_POST_SPIKE_CODE("postTrace += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("preTrace *= tauPlusDecay;\n");
    SET_POST_DYNAMICS_CODE("postTrace *= tauMinusDecay;\n");
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

    STDP stdpCopy;
    ASSERT_EQ(STDP::getInstance()->getHashDigest(), stdpCopy.getHashDigest());
}
//--------------------------------------------------------------------------
TEST(WeightUpdateModels, ValidateParamValues) 
{
    const VarValues varVals{{"g", uninitialisedVar()}};
    const VarValues preVarVals{{"preTrace", uninitialisedVar()}};
    const VarValues postVarVals{{"postTrace", uninitialisedVar()}};
    const LocalVarReferences preNeuronVarRefs{};
    const LocalVarReferences postNeuronVarRefs{};
    const LocalVarReferences psmVarRefs{};

    const ParamValues paramValsCorrect{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const ParamValues paramValsMisSpelled{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"APlus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const ParamValues paramValsMissing{{"tauPlus", 10.0}, {"tauMinus", 10.0},{"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const ParamValues paramValsExtra{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Bminus", 0.01},{"Wmin", 0.0}, {"Wmax", 1.0}};

    STDPAdditive::getInstance()->validate(paramValsCorrect, varVals, preVarVals, postVarVals, 
                                          preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);

    try {
        STDPAdditive::getInstance()->validate(paramValsMisSpelled, varVals, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramValsMissing, varVals, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramValsExtra, varVals, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
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
    const LocalVarReferences preNeuronVarRefs{};
    const LocalVarReferences postNeuronVarRefs{};
    const LocalVarReferences psmVarRefs{};
    
    const VarValues varValsCorrect{{"g", uninitialisedVar()}};
    const VarValues varValsMisSpelled{{"G", uninitialisedVar()}};
    const VarValues varValsMissing{};
    const VarValues varValsExtra{{"g", uninitialisedVar()}, {"d", uninitialisedVar()}};

    STDPAdditive::getInstance()->validate(paramVals, varValsCorrect, preVarVals, postVarVals, 
                                          preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsMisSpelled, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsMissing, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varValsExtra, preVarVals, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
//--------------------------------------------------------------------------
TEST(WeightUpdateModels, ValidatePreVarValues) 
{
    const VarValues postVarVals{{"postTrace", 0.0}};
    const VarValues varVals{{"g", uninitialisedVar()}};
    const ParamValues paramVals{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const LocalVarReferences preNeuronVarRefs{};
    const LocalVarReferences postNeuronVarRefs{};
    const LocalVarReferences psmVarRefs{};
    
    const VarValues preVarValsCorrect{{"preTrace", 0.0}};
    const VarValues preVarValsMisSpelled{{"prETrace", 0.0}};
    const VarValues preVarValsMissing{};
    const VarValues preVarValsExtra{{"preTrace", 0.0}, {"postTrace", 0.0}};

    STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsCorrect, postVarVals, 
                                          preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsMisSpelled, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsMissing, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarValsExtra, postVarVals, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
//--------------------------------------------------------------------------
TEST(WeightUpdateModels, ValidatePostVarValues) 
{
    const VarValues preVarVals{{"preTrace", 0.0}};
    const VarValues varVals{{"g", uninitialisedVar()}};
    const ParamValues paramVals{{"tauPlus", 10.0}, {"tauMinus", 10.0}, {"Aplus", 0.01}, {"Aminus", 0.01}, {"Wmin", 0.0}, {"Wmax", 1.0}};
    const LocalVarReferences preNeuronVarRefs{};
    const LocalVarReferences postNeuronVarRefs{};
    const LocalVarReferences psmVarRefs{};
    
    const VarValues postVarValsCorrect{{"postTrace", 0.0}};
    const VarValues postVarValsMisSpelled{{"PostTrace", 0.0}};
    const VarValues postVarValsMissing{};
    const VarValues postVarValsExtra{{"postTrace", 0.0}, {"preTrace", 0.0}};

    STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsCorrect, 
                                          preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsMisSpelled, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsMissing, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        STDPAdditive::getInstance()->validate(paramVals, varVals, preVarVals, postVarValsExtra, 
                                              preNeuronVarRefs, postNeuronVarRefs, psmVarRefs);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
