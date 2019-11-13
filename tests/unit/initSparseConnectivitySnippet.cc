// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "initSparseConnectivitySnippet.h"

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
