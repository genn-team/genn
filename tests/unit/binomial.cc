// Standard C includes
#include <cmath>

// Google test includes
#include "gtest/gtest.h"

// GeNN code generator includes
#include "binomial.h"

using namespace GeNN;

//--------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------
TEST(Binomial, FixedProbabilityRange)
{
    const double probs[] = {0.02, 0.1, 1.0};
    const unsigned int popSizes[] = {1, 1000, 1000000};
    constexpr size_t numProbs = sizeof(probs) / sizeof(probs[0]);
    constexpr size_t numPopSizes = sizeof(popSizes) / sizeof(popSizes[0]);

    const unsigned int scipyPPF[numProbs][numPopSizes] = {
        // popSizes[0], popSizes[1], popSizes[2]
        {1,             47,         20897},     // probs[0] 
        {1,             153,        101914},    // probs[1]
        {1,             1000,       1000000}};  // probs[2]

    for(size_t i = 0; i < numProbs; i++) {
        for(size_t j = 0; j < numPopSizes; j++) {
            const double quantile = pow(0.9999, 1.0 / (double)popSizes[j]);
            const double ppf = binomialInverseCDF(quantile, popSizes[j], probs[i]);
            EXPECT_EQ(scipyPPF[i][j], ppf);
        }
    }
}

TEST(Binomial, FixedNumberPrePost)
{
    const unsigned int fixedNumbers[] = {1, 10, 100};
    const unsigned int popSizes[] = {1, 1000, 1000000};
    constexpr size_t numFixedNumbers = sizeof(fixedNumbers) / sizeof(fixedNumbers[0]);
    constexpr size_t numPopSizes = sizeof(popSizes) / sizeof(popSizes[0]);

    const unsigned int scipyPPF[numFixedNumbers][numPopSizes] = {
        // popSizes[0], popSizes[1], popSizes[2]
        {1,             10,         12},     // fixedNumbers[0]
        {10,            30,         36},     // fixedNumbers[1]
        {100,           156,        170}};   // fixedNumbers[2]

    for(size_t i = 0; i < numFixedNumbers; i++) {
        for(size_t j = 0; j < numPopSizes; j++) {
            if(fixedNumbers[i] < popSizes[j]) {
                const double quantile = pow(0.9999, 1.0 / (double)popSizes[j]);
                const double ppf = binomialInverseCDF(quantile, fixedNumbers[i] * popSizes[j], 1.0 / (double)popSizes[j]);
                EXPECT_EQ(scipyPPF[i][j], ppf);
            }
        }
    }
}
