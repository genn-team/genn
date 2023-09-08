// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"
#include "neuronModels.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// LIFCopy
//--------------------------------------------------------------------------
class LIFCopy : public NeuronModels::Base
{
public:
    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0) {\n"
        "  scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else {\n"
        "  $(RefracTime) -= dt;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauM")); }},
        {"Rmembrane", [](const ParamValues &pars, double){ return  pars.at("TauM") / pars.at("C"); }}});

    SET_NEURON_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(NeuronModels, CompareBuiltIn)
{
    ASSERT_EQ(NeuronModels::LIF::getInstance()->getHashDigest(), NeuronModels::LIF::getInstance()->getHashDigest());
    ASSERT_NE(NeuronModels::LIF::getInstance()->getHashDigest(), NeuronModels::Izhikevich::getInstance()->getHashDigest());
    ASSERT_NE(NeuronModels::Izhikevich::getInstance()->getHashDigest(), NeuronModels::IzhikevichVariable::getInstance()->getHashDigest());
    ASSERT_NE(NeuronModels::TraubMilesAlt::getInstance()->getHashDigest(), NeuronModels::TraubMiles::getInstance()->getHashDigest());
}

TEST(NeuronModels, CompareCopyPasted)
{
    using namespace NeuronModels;

    LIFCopy lifCopy;
    ASSERT_EQ(NeuronModels::LIF::getInstance()->getHashDigest(), lifCopy.getHashDigest());
}
//--------------------------------------------------------------------------
TEST(NeuronModels, ValidateParamValues) 
{
    const VarValues varVals{{"V", uninitialisedVar()}, {"U", uninitialisedVar()}};

    const std::unordered_map<std::string, double> paramValsCorrect{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    const std::unordered_map<std::string, double> paramValsMisSpelled{{"a", 0.02}, {"B", 0.2}, {"c", -65.0}, {"d", 8.0}};
    const std::unordered_map<std::string, double> paramValsMissing{{"a", 0.02}, {"b", 0.2}, {"d", 8.0}};
    const std::unordered_map<std::string, double> paramValsExtra{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}, {"e", 8.0}};

    NeuronModels::Izhikevich::getInstance()->validate(paramValsCorrect, varVals, "Neuron group");

    try {
        NeuronModels::Izhikevich::getInstance()->validate(paramValsMisSpelled, varVals, "Neuron group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        NeuronModels::Izhikevich::getInstance()->validate(paramValsMissing, varVals, "Neuron group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        NeuronModels::Izhikevich::getInstance()->validate(paramValsExtra, varVals, "Neuron group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
//--------------------------------------------------------------------------
TEST(NeuronModels, ValidateVarValues) 
{
    const std::unordered_map<std::string, double> paramVals{{"a", 0.02}, {"b", 0.2}, {"c", -65.0}, {"d", 8.0}};
    
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsCorrect{{"V", 0.0}, {"U", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsMisSpelled{{"V", 0.0}, {"u", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsMissing{{"V", 0.0}};
    const std::unordered_map<std::string, InitVarSnippet::Init> varValsExtra{{"V", 0.0}, {"U", 0.0}, {"A", 0.0}};

    NeuronModels::Izhikevich::getInstance()->validate(paramVals, varValsCorrect, "Neuron group");

    try {
        NeuronModels::Izhikevich::getInstance()->validate(paramVals, varValsMisSpelled, "Neuron group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        NeuronModels::Izhikevich::getInstance()->validate(paramVals, varValsMissing, "Neuron group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        NeuronModels::Izhikevich::getInstance()->validate(paramVals, varValsExtra, "Neuron group");
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
