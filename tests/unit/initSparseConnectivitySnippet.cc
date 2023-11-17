// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "modelSpec.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// OneToOneCopy
//--------------------------------------------------------------------------
class OneToOneCopy : public InitSparseConnectivitySnippet::Base
{
public:
    SET_ROW_BUILD_CODE(
        "addSynapse(id_pre);\n");

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
    DECLARE_SNIPPET(FixedNumberTotalWithReplacement);

     SET_ROW_BUILD_CODE(
        "scalar x = 0.0;\n"
        "for(unsigned int c = 0; c < preCalcRowLength[(id_pre * num_threads) + id_thread]; c++) {\n"
        "   const scalar u = gennrand_uniform();\n"
        "   x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)(rowLength - c)));\n"
        "   unsigned int postIdx = (unsigned int)(x * num_post);\n"
        "   postIdx = (postIdx < num_post) ? postIdx : (num_post - 1);\n"
        "   addSynapse(postIdx + id_post_begin);\n"
        "}\n");
    SET_PARAM_NAMES({"total"});
    SET_EXTRA_GLOBAL_PARAMS({{"preCalcRowLength", "unsigned int*"}})

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const ParamValues &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars.at("total").cast<unsigned int>(), (double)numPost / ((double)numPre * (double)numPost));
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const ParamValues &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars.at("total").cast<unsigned int>(), (double)numPre / ((double)numPre * (double)numPost));
        });
};
IMPLEMENT_SNIPPET(FixedNumberTotalWithReplacement);

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(InitSparseConnectivitySnippet, CompareBuiltIn)
{
    using namespace InitSparseConnectivitySnippet;

    ASSERT_EQ(OneToOne::getInstance()->getHashDigest(), OneToOne::getInstance()->getHashDigest());
    ASSERT_NE(OneToOne::getInstance()->getHashDigest(), FixedProbability::getInstance()->getHashDigest());
    ASSERT_NE(FixedProbability::getInstance()->getHashDigest(), FixedProbabilityNoAutapse::getInstance()->getHashDigest());
}

TEST(InitSparseConnectivitySnippet, CompareCopyPasted)
{
    using namespace InitSparseConnectivitySnippet;

    OneToOneCopy oneToOneCopy;
    ASSERT_EQ(OneToOne::getInstance()->getHashDigest(), oneToOneCopy.getHashDigest());
}

TEST(InitSparseConnectivitySnippet, CompareVarInitParameters)
{
    ParamValues fixedProbParamsA{{"prob", 0.1}};
    ParamValues fixedProbParamsB{{"prob", 0.4}};
    auto connectivityInit0 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA);
    auto connectivityInit1 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA);
    auto connectivityInit2 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsB);

    connectivityInit0.finalise(0.1);
    connectivityInit1.finalise(0.1);
    connectivityInit2.finalise(0.1);

    ASSERT_EQ(connectivityInit0.getHashDigest(), connectivityInit1.getHashDigest());
    ASSERT_EQ(connectivityInit0.getHashDigest(), connectivityInit2.getHashDigest());
}

TEST(InitSparseConnectivitySnippet, CompareUnusedParameters)
{
    ParamValues fixedNumberParamsA{{"total", 1000}};
    ParamValues fixedNumberParamsB{{"total", 1200}};

    auto connectivityInit0 = initConnectivity<FixedNumberTotalWithReplacement>(fixedNumberParamsA);
    auto connectivityInit1 = initConnectivity<FixedNumberTotalWithReplacement>(fixedNumberParamsB);

    connectivityInit0.finalise(0.1);
    connectivityInit1.finalise(0.1);

    ASSERT_EQ(connectivityInit0.getHashDigest(), connectivityInit1.getHashDigest());
}
//--------------------------------------------------------------------------
TEST(InitSparseConnectivitySnippet, ValidateParamValues) 
{
    const ParamValues paramValsCorrect{{"prob", 0.1}};
    const ParamValues paramValsMisSpelled{{"probs", 0.1}};
    const ParamValues paramValsMissing{};
    const ParamValues paramValsExtra{{"prob", 0.1}, {"sd", 1.0}};

    InitSparseConnectivitySnippet::FixedProbability::getInstance()->validate(paramValsCorrect);

    try {
        InitSparseConnectivitySnippet::FixedProbability::getInstance()->validate(paramValsMisSpelled);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        InitSparseConnectivitySnippet::FixedProbability::getInstance()->validate(paramValsMissing);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 

    try {
        InitSparseConnectivitySnippet::FixedProbability::getInstance()->validate(paramValsExtra);
        FAIL();
    }
    catch(const std::runtime_error &) {
    } 
}
