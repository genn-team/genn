// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"

//--------------------------------------------------------------------------
// OneToOneCopy
//--------------------------------------------------------------------------
class OneToOneCopy : public InitSparseConnectivitySnippet::Base
{
public:
    SET_ROW_BUILD_CODE(
        "$(addSynapse, $(id_pre));\n"
        "$(endRow);\n");

    SET_MAX_ROW_LENGTH(1);
    SET_MAX_COL_LENGTH(1);
};


//----------------------------------------------------------------------------
// FixedNumberTotalWithReplacement
//----------------------------------------------------------------------------
//! Initialises variable by sampling from the uniform distribution
class FixedNumberTotalWithReplacement : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(FixedNumberTotalWithReplacement, 1);

    SET_ROW_BUILD_CODE(
        "const unsigned int rowLength = $(preCalcRowLength)[($(id_pre) * $(num_threads)) + $(id_thread)];\n"
        "if(c >= rowLength) {\n"
        "   $(endRow);\n"
        "}\n"
        "const scalar u = $(gennrand_uniform);\n"
        "x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)(rowLength - c)));\n"
        "unsigned int postIdx = (unsigned int)(x * $(num_post));\n"
        "postIdx = (postIdx < $(num_post)) ? postIdx : ($(num_post) - 1);\n"
        "$(addSynapse, postIdx + $(id_post_begin));\n"
        "c++;\n");
    SET_ROW_BUILD_STATE_VARS({{"x", "scalar", 0.0},{"c", "unsigned int", 0}});

    SET_PARAM_NAMES({"total"});
    SET_EXTRA_GLOBAL_PARAMS({{"preCalcRowLength", "unsigned int*"}})

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, (unsigned int)pars[0], (double)numPost / ((double)numPre * (double)numPost));
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, (unsigned int)pars[0], (double)numPre / ((double)numPre * (double)numPost));
        });
};
IMPLEMENT_SNIPPET(FixedNumberTotalWithReplacement);

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(InitSparseConnectivitySnippet, CompareBuiltIn)
{
    ASSERT_TRUE(InitSparseConnectivitySnippet::OneToOne::getInstance()->canBeMerged(InitSparseConnectivitySnippet::OneToOne::getInstance()));
    ASSERT_FALSE(InitSparseConnectivitySnippet::OneToOne::getInstance()->canBeMerged(InitSparseConnectivitySnippet::FixedProbability::getInstance()));
    ASSERT_FALSE(InitSparseConnectivitySnippet::FixedProbability::getInstance()->canBeMerged(InitSparseConnectivitySnippet::FixedProbabilityNoAutapse::getInstance()));
}

TEST(InitSparseConnectivitySnippet, CompareCopyPasted)
{
    OneToOneCopy oneToOneCopy;
    ASSERT_TRUE(InitSparseConnectivitySnippet::OneToOne::getInstance()->canBeMerged(&oneToOneCopy));
}

TEST(InitSparseConnectivitySnippet, CompareVarInitParameters)
{
    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProbParamsA(0.1);
    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProbParamsB(0.4);

    auto connectivityInit0 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA);
    auto connectivityInit1 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA);
    auto connectivityInit2 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsB);

    connectivityInit0.initDerivedParams(0.1);
    connectivityInit1.initDerivedParams(0.1);
    connectivityInit2.initDerivedParams(0.1);

    ASSERT_TRUE(connectivityInit0.canBeMerged(connectivityInit1));
    ASSERT_TRUE(connectivityInit0.canBeMerged(connectivityInit2));
}

TEST(InitSparseConnectivitySnippet, CompareUnusedParameters)
{
    FixedNumberTotalWithReplacement::ParamValues fixedNumberParamsA(1000);
    FixedNumberTotalWithReplacement::ParamValues fixedNumberParamsB(1200);

    auto connectivityInit0 = initConnectivity<FixedNumberTotalWithReplacement>(fixedNumberParamsA);
    auto connectivityInit1 = initConnectivity<FixedNumberTotalWithReplacement>(fixedNumberParamsB);

    connectivityInit0.initDerivedParams(0.1);
    connectivityInit1.initDerivedParams(0.1);

    ASSERT_TRUE(connectivityInit0.canBeMerged(connectivityInit1));
}
