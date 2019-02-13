#pragma once

// Standard C include
#include <cmath>

// Standard C++ includes
#include <numeric>

//----------------------------------------------------------------------------
// SimulationSynapsePolicyNone
//----------------------------------------------------------------------------
class SimulationSynapsePolicyNone
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void Init()
    {
    }

    template<typename UpdateFn, typename StepGeNNFn>
    float Simulate(UpdateFn updateFn, StepGeNNFn stepGeNNFn)
    {
        float err = 0.0f;
        inputpre = 0.0f;
        while(t < 20.0f) {
            // for all pre-synaptic neurons
            float x[10];
            for (int j = 0; j < 10; j++) {
                // generate expected values
                float newX;
                if(updateFn(iT, j, t, newX)) {
                    x[j]= newX;
                }
                else {
                    x[j] = 0.0f;
                }
            }

            // Add error for this time step to total
            err += std::inner_product(&x[0], &x[10],
                                      &xpre[glbSpkShiftpre],
                                      0.0f,
                                      std::plus<float>(),
                                      [](float a, float b){ return std::fabs(a - b); });

            // Update global
            inputpre = std::pow(t, 2.0f);

            // Step GeNN kernel
            stepGeNNFn();
        }

        return err;
    }
};
