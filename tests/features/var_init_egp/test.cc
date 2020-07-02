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

#define CONFIGURE_REPEAT_EGP(NAME)          \
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
        
        // Set scalar egps
        valvconstantPop =  7.0f;
        valvconstantCurrSource =  7.0f;
        valpvconstantSparse =  7.0f;
        valpvconstantDense =  7.0f;
        valvconstantSparse =  7.0f;
        valvconstantDense =  7.0f;
        valpre_vconstantSparse =  7.0f;
        valpre_vconstantDense =  7.0f;
        valpost_vconstantSparse =  7.0f;
        valpost_vconstantDense =  7.0f;
        
        // Configure EGP arrays containing repeating pattern
        CONFIGURE_REPEAT_EGP(valuesvrepeatPop);
        CONFIGURE_REPEAT_EGP(valuesvrepeatCurrSource);
        CONFIGURE_REPEAT_EGP(valuespvrepeatSparse);
        CONFIGURE_REPEAT_EGP(valuespvrepeatDense);
        CONFIGURE_REPEAT_EGP(valuesvrepeatSparse);
        CONFIGURE_REPEAT_EGP(valuesvrepeatDense);
        CONFIGURE_REPEAT_EGP(valuespre_vrepeatSparse);
        CONFIGURE_REPEAT_EGP(valuespre_vrepeatDense);
        CONFIGURE_REPEAT_EGP(valuespost_vrepeatSparse);
        CONFIGURE_REPEAT_EGP(valuespost_vrepeatDense);
        initialize();
        
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
        // Check values initialised using repeating values are correct
        const float correctRepeatVal = (float)(i % 10);
        EXPECT_EQ(vrepeatPop[i], correctRepeatVal);
        EXPECT_EQ(vrepeatCurrSource[i], correctRepeatVal);
        EXPECT_EQ(pvrepeatSparse[i], correctRepeatVal);
        EXPECT_EQ(pvrepeatDense[i], correctRepeatVal);
        EXPECT_EQ(vrepeatSparse[i], correctRepeatVal);
        EXPECT_EQ(vrepeatDense[i], correctRepeatVal);
        EXPECT_EQ(pre_vrepeatSparse[i], correctRepeatVal);
        EXPECT_EQ(post_vrepeatSparse[i], correctRepeatVal);
        EXPECT_EQ(post_vrepeatDense[i], correctRepeatVal);
        
        // Check values initialised using constnat values are correct
        EXPECT_EQ(vconstantPop[i], 7.0f);
        EXPECT_EQ(vconstantCurrSource[i], 7.0f);
        EXPECT_EQ(pvconstantSparse[i], 7.0f);
        EXPECT_EQ(pvconstantDense[i], 7.0f);
        EXPECT_EQ(vconstantSparse[i], 7.0f);
        EXPECT_EQ(vconstantDense[i], 7.0f);
        EXPECT_EQ(pre_vconstantSparse[i], 7.0f);
        EXPECT_EQ(post_vconstantSparse[i], 7.0f);
        EXPECT_EQ(post_vconstantDense[i], 7.0f);
    }
    
}

