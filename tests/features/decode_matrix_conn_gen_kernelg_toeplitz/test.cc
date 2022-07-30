#include <algorithm>
#include <iostream>

// Auto-generated simulation code includess
#include "decode_matrix_conn_gen_kernelg_toeplitz_CODE/definitions.h"

// **NOTE** base-class for simulation tests must be
// included after auto-generated globals are includes
#include "../../utils/simulation_test.h"
#include "../../utils/conv_data.h"


//----------------------------------------------------------------------------
// SimTest
//----------------------------------------------------------------------------
class SimTest : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
        // Allocate, configure and upload (normalised) vertical sobel convolution kernel
        gSynVert[0] = 1.0f;   gSynVert[1] = 0.0f;   gSynVert[2] = -1.0f;
        gSynVert[3] = 2.0f;   gSynVert[4] = 0.0f;   gSynVert[5] = -2.0f;
        gSynVert[6] = 1.0f;   gSynVert[7] = 0.0f;   gSynVert[8] = -1.0f;

        // Allocate, configure and upload (normalised) horizontal sobel convolution kernel
        gSynHorz[0] = 1.0f;   gSynHorz[1] = 2.0f;   gSynHorz[2] = 1.0f;
        gSynHorz[3] = 0.0f;   gSynHorz[4] = 0.0f;   gSynHorz[5] = -0.0f;
        gSynHorz[6] = -1.0f;  gSynHorz[7] = -2.0f;  gSynHorz[8] = -1.0f;
    }


    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        // Set input spikes and push
        const size_t numSpikes = sizeof(testPattern) / sizeof(unsigned int);
        spikeCount_Pre = (unsigned int)numSpikes;
        for(size_t i = 0; i < numSpikes; i++) {
            spike_Pre[i] = testPattern[i];
        }
        pushPreCurrentSpikesToDevice();
        
        // Simulate one timestep
        stepTime();
        
        // Pull output
        pullxPostVertFromDevice();
        pullxPostHorzFromDevice();
        
        // Check that vertical output is correct
        if(!std::equal(&xPostVert[0], &xPostVert[62 * 62], verticalOutput,
                       [](float a, float b){ return std::abs(a - b) < std::numeric_limits<float>::epsilon(); }))
        {
            return false;
        }
        
        // Check that horizontal output is correct
        if(!std::equal(&xPostHorz[0], &xPostHorz[62 * 62], horizontalOutput,
                       [](float a, float b){ return std::abs(a - b) < std::numeric_limits<float>::epsilon(); }))
        {
            return false;
        }
        
        return true;
    }
};

TEST_F(SimTest, DecodeMatrixConnGenProceduralgProceduralKernel)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}
