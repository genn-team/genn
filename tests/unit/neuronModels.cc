// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "neuronModels.h"

// Unit test includes
#include "hashUtils.h"

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
    ASSERT_TRUE(NeuronModels::LIF::getInstance()->canBeMerged(NeuronModels::LIF::getInstance()));
    ASSERT_FALSE(NeuronModels::LIF::getInstance()->canBeMerged(NeuronModels::Izhikevich::getInstance()));
    ASSERT_FALSE(NeuronModels::Izhikevich::getInstance()->canBeMerged(NeuronModels::IzhikevichVariable::getInstance()));
    ASSERT_FALSE(NeuronModels::TraubMilesAlt::getInstance()->canBeMerged(NeuronModels::TraubMiles::getInstance()));

    ASSERT_HASH_EQ(NeuronModels::LIF::getInstance(), NeuronModels::LIF::getInstance(), updateHash);
    ASSERT_HASH_NE(NeuronModels::LIF::getInstance(), NeuronModels::Izhikevich::getInstance(), updateHash);
    ASSERT_HASH_NE(NeuronModels::Izhikevich::getInstance(), NeuronModels::IzhikevichVariable::getInstance(), updateHash);
    ASSERT_HASH_NE(NeuronModels::TraubMilesAlt::getInstance(), NeuronModels::TraubMiles::getInstance(), updateHash);
}

TEST(NeuronModels, CompareCopyPasted)
{
    using namespace NeuronModels;

    LIFCopy lifCopy;
    ASSERT_TRUE(NeuronModels::LIF::getInstance()->canBeMerged(&lifCopy));

    ASSERT_HASH_EQ(NeuronModels::LIF::getInstance(), &lifCopy, updateHash);
}
