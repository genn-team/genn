#pragma once

// Standard C includes
#include <cmath>

// Standard C++ includes
#include <functional>
#include <numeric>

//----------------------------------------------------------------------------
// SimulationSynapsePolicyDense
//----------------------------------------------------------------------------
class SimulationSynapsePolicyDense
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void Init()
    {
        // Create array pointing to weights for each synapse group
        m_TheW[0] = wsyn0;
        m_TheW[1] = wsyn1;
        m_TheW[2] = wsyn2;
        m_TheW[3] = wsyn3;
        m_TheW[4] = wsyn4;
        m_TheW[5] = wsyn5;
        m_TheW[6] = wsyn6;
        m_TheW[7] = wsyn7;
        m_TheW[8] = wsyn8;
        m_TheW[9] = wsyn9;
    }

    template<typename UpdateFn, typename StepGeNNFn>
    float Simulate(UpdateFn updateFn, StepGeNNFn stepGeNNFn)
    {
        float err = 0.0f;
        float x[10][100];
        while(t < 20.0f) {
            // for each delay
            for (int d = 0; d < 10; d++) {
                // for all pre-synaptic neurons
                for (int j = 0; j < 10; j++) {
                    // for all post-syn neurons
                    for (int k = 0; k < 10; k++) {
                        float newX;
                        if(updateFn(d, j, k, t, newX)) {
                            x[d][(j * 10) + k] = newX;
                        }
                        else if(iT == 0) {
                            x[d][(j * 10) + k] = 0.0f;
                        }
                    }
                }

                // Add error for this time step to total
                err += std::inner_product(&x[d][0], &x[d][100],
                                          GetTheW(d),
                                          0.0f,
                                          std::plus<float>(),
                                          [](float a, float b){ return std::fabs(a - b); });
            }

            // Step GeNN kernel
            stepGeNNFn();
        }

        return err;
    }

protected:
    //--------------------------------------------------------------------------
    // Protected API
    //--------------------------------------------------------------------------
    float *GetTheW(unsigned int delay) const
    {
        return m_TheW[delay];
    }

private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    float *m_TheW[10];
};
