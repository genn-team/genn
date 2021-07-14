// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "neuronModels.h"

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
        "  $(RefracTime) -= DT;\n"
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
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});

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
