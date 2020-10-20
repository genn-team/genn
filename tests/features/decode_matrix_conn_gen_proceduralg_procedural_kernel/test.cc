#include <algorithm>
#include <iostream>

// Auto-generated simulation code includess
#include "decode_matrix_conn_gen_proceduralg_procedural_kernel_CODE/definitions.h"

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
    virtual void PreInit()
    {
        // Allocate, configure and upload (normalised) vertical sobel convolution kernel
        allocatekernelgSynVert(3 * 3);
        kernelgSynVert[0] = 1.0f;   kernelgSynVert[1] = 0.0f;   kernelgSynVert[2] = -1.0f;
        kernelgSynVert[3] = 2.0f;   kernelgSynVert[4] = 0.0f;   kernelgSynVert[5] = -2.0f;
        kernelgSynVert[6] = 1.0f;   kernelgSynVert[7] = 0.0f;   kernelgSynVert[8] = -1.0f;
        pushkernelgSynVertToDevice(3 * 3);

        // Allocate, configure and upload (normalised) horizontal sobel convolution kernel
        allocatekernelgSynHorz(3 * 3);
        kernelgSynHorz[0] = 1.0f;   kernelgSynHorz[1] = 2.0f;   kernelgSynHorz[2] = 1.0f;
        kernelgSynHorz[3] = 0.0f;   kernelgSynHorz[4] = 0.0f;   kernelgSynHorz[5] = -0.0f;
        kernelgSynHorz[6] = -1.0f;  kernelgSynHorz[7] = -2.0f;  kernelgSynHorz[8] = -1.0f;
        pushkernelgSynHorzToDevice(3 * 3);
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
