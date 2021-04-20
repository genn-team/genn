#include <algorithm>
#include <iostream>

// Auto-generated simulation code includess
#include "decode_matrix_conn_gen_individualg_ragged_kernel_CODE/definitions.h"

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
        // Allocate, configure and upload convolution kernels
        allocatekernelgSyn(3 * 3 * 2);
        configureKernel(kernelgSyn);
        pushkernelgSynToDevice(3 * 3 * 2);
        
        // Allocate, configure and upload convolution kernels
        allocatekernelgSynPool(3 * 3 * 2);
        configureKernel(kernelgSynPool, 0.25f);
        pushkernelgSynPoolToDevice(3 * 3 * 2);
    }


    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        // Set input spikes 
        const size_t numSpikes = sizeof(testPattern) / sizeof(unsigned int);
        spikeCount_Pre = (unsigned int)numSpikes;
        spikeCount_PrePool = 4 * (unsigned int)numSpikes;
        for(size_t s = 0; s < numSpikes; s++) {
            // Copy spike directly into non-pooled spike source
            spike_Pre[s] = testPattern[s];
            
            // Convert spike index into rows and columns
            const unsigned int i = testPattern[s] / 64;
            const unsigned int j = testPattern[s] % 64;
            
            // Add a spike to each pooled inputs
            spike_PrePool[(s * 4) + 0] = ((i * 2) * 128) + (j * 2);
            spike_PrePool[(s * 4) + 1] = ((i * 2) * 128) + ((j * 2) + 1);
            spike_PrePool[(s * 4) + 2] = (((i * 2) + 1) * 128) + ((j * 2) + 1);
            spike_PrePool[(s * 4) + 3] = (((i * 2) + 1) * 128) + (j * 2);
        }
        
        
        // Push spikes to device
        pushPreCurrentSpikesToDevice();
        pushPrePoolCurrentSpikesToDevice();
        
        // Simulate one timestep
        stepTime();
        
        // Pull output
        pullxPostFromDevice();
        pullxPostPoolFromDevice();

        // Verify
        if(!verifyOutput(xPost)) {
            return false;
        }
        if(!verifyOutput(xPostPool)) {
            return false;
        }

        return true;
    }

private:
    void configureKernel(float *kernel, float scale = 1.0f)
    {
        // Configure (normalised) vertical sobel convolution kernel
        kernel[0] = 1.0f * scale;   kernel[2] = 0.0f;   kernel[4] = -1.0f * scale;
        kernel[6] = 2.0f * scale;   kernel[8] = 0.0f;   kernel[10] = -2.0f * scale;
        kernel[12] = 1.0f * scale;  kernel[14] = 0.0f;  kernel[16] = -1.0f * scale;
        

        // Configure  (normalised) horizontal sobel convolution kernel
        kernel[1] = 1.0f * scale;   kernel[3] = 2.0f * scale;   kernel[5] = 1.0f * scale;
        kernel[7] = 0.0f;           kernel[9] = 0.0f;           kernel[11] = -0.0f;
        kernel[13] = -1.0f * scale; kernel[15] = -2.0f * scale; kernel[17] = -1.0f * scale;
    }
    
    bool verifyOutput(float *output)
    {
        // Validate output
        for(unsigned int i = 0; i < 62; i++) {
            for(unsigned int j = 0; j < 62; j++) {
                if(output[(i * 62 * 2) + (j * 2)] != verticalOutput[(i * 62) + j]) { 
                    std::cout << output[(i * 62 * 2) + (j * 2)] << "!=" << verticalOutput[(i * 62) + j] << std::endl;
                    return false;
                }
                if(output[(i * 62 * 2) + (j * 2) + 1] != horizontalOutput[(i * 62) + j]) { 
                    std::cout << output[(i * 62 * 2) + (j * 2) + 1] << "!=" << horizontalOutput[(i * 62) + j] << std::endl;
                    return false;
                }
            }
        }
        
        return true;
    }
};

TEST_F(SimTest, DecodeMatrixConnGenIndividualgRaggedKernel)
{
    // Check total error is less than some tolerance
    EXPECT_TRUE(Simulate());
}
