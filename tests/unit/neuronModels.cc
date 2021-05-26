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
    using namespace NeuronModels;

    ASSERT_TRUE(LIF::getInstance()->canBeMerged(LIF::getInstance()));
    ASSERT_FALSE(LIF::getInstance()->canBeMerged(Izhikevich::getInstance()));
    ASSERT_FALSE(Izhikevich::getInstance()->canBeMerged(IzhikevichVariable::getInstance()));
    ASSERT_FALSE(TraubMilesAlt::getInstance()->canBeMerged(TraubMiles::getInstance()));

    {
        boost::uuids::detail::sha1 a;
        updateHash(*LIF::getInstance(), a);

        boost::uuids::detail::sha1::digest_type aDigest;
        a.get_digest(aDigest);

        boost::uuids::detail::sha1 b;
        updateHash(*LIF::getInstance(), b);

        boost::uuids::detail::sha1::digest_type bDigest;
        b.get_digest(bDigest);

        ASSERT_TRUE(std::equal(std::begin(aDigest), std::end(aDigest), std::begin(bDigest)));
    }
    /*ASSERT_HASH_EQ(Base, *LIF::getInstance(), *LIF::getInstance());
    ASSERT_HASH_NE(Base, *LIF::getInstance(), *Izhikevich::getInstance());
    ASSERT_HASH_NE(Base, *Izhikevich::getInstance(), *IzhikevichVariable::getInstance());
    ASSERT_HASH_NE(Base, *TraubMilesAlt::getInstance(), *TraubMiles::getInstance());*/
}

TEST(NeuronModels, CompareCopyPasted)
{
    using namespace NeuronModels;

    LIFCopy lifCopy;
    ASSERT_TRUE(LIF::getInstance()->canBeMerged(&lifCopy));

    //ASSERT_HASH_EQ(Base, lifCopy, *LIF::getInstance());
}
