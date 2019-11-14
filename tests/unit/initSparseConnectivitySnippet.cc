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

    const auto connectivityInit0 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA);
    const auto connectivityInit1 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsA);
    const auto connectivityInit2 = initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProbParamsB);

    ASSERT_TRUE(connectivityInit0.canBeMerged(connectivityInit1));
    ASSERT_FALSE(connectivityInit0.canBeMerged(connectivityInit2));
}