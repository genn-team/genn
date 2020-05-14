//--------------------------------------------------------------------------
/*! \file var_init/test.cc

\brief Main test code that is part of the feature testing
suite of minimal models with known analytic outcomes that are used for continuous integration testing.
*/
//--------------------------------------------------------------------------
#include <numeric>

// Google test includes
#include "gtest/gtest.h"

// Auto-generated simulation code includess
#include "var_init_egp_CODE/definitions.h"

#define CONFIGURE_EGP(NAME)                 \
    allocate##NAME(10);                     \
    std::iota(&NAME[0], &NAME[10], 0.0f);   \
    push##NAME##ToDevice(10)

//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public ::testing::Test
{
protected:
    //--------------------------------------------------------------------------
    // test virtuals
    //--------------------------------------------------------------------------
    virtual void SetUp() override
    {
        // Perform GeNN initialization
        allocateMem();
        
        // Allocate EGPs
        CONFIGURE_EGP(valuesvPop);
        CONFIGURE_EGP(valuesvCurrSource);
        CONFIGURE_EGP(valuespvSparse);
        CONFIGURE_EGP(valuespvDense);
        CONFIGURE_EGP(valuesvSparse);
        CONFIGURE_EGP(valuesvDense);
        CONFIGURE_EGP(valuespre_vSparse);
        CONFIGURE_EGP(valuespre_vDense);
        CONFIGURE_EGP(valuespost_vSparse);
        CONFIGURE_EGP(valuespost_vDense);
        initialize();
        
        // Build sparse connectors
        rowLengthSparse[0] = 100;
        for(unsigned int i = 0; i < 100; i++) {
            indSparse[i] = i;
        }
        
        initializeSparse();
    }

    virtual void TearDown() override
    {
        freeMem();
    }
};

TEST_F(SimTest, VarInitEGP)
{
    // Pull vars back to host
    pullPopStateFromDevice();
    pullCurrSourceStateFromDevice();
    pullDenseStateFromDevice();
    pullSparseStateFromDevice();
    
    for(size_t i = 0; i < 100; i++) {
        const float correct = (float)(i % 10);
        EXPECT_EQ(vPop[i], correct);
        EXPECT_EQ(vCurrSource[i], correct);
        EXPECT_EQ(pvSparse[i], correct);
        EXPECT_EQ(pvDense[i], correct);
        EXPECT_EQ(vSparse[i], correct);
        EXPECT_EQ(vDense[i], correct);
        EXPECT_EQ(pre_vSparse[i], correct);
        EXPECT_EQ(pre_vDense[i], correct);
        EXPECT_EQ(post_vSparse[i], correct);
        EXPECT_EQ(post_vDense[i], correct);
    }
    
}

