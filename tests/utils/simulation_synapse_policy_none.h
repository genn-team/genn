#pragma once

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
        for (int i = 0; i < (int)(20.0f / DT); i++)
        {
            // **YUCK** update global time - this shouldn't be user responsibility
            t = i * DT;

            // for all pre-synaptic neurons
            float x[10];
            for (int j = 0; j < 10; j++)
            {
                // generate expected values
                float newX;
                if(updateFn(i, j, t, newX))
                {
                    x[j]= newX;
                }
                else
                {
                    x[j] = 0.0f;
                }
            }

            // Add error for this time step to total
            err += std::inner_product(&x[0], &x[10],
                                      &xpre[glbSpkShiftpre],
                                      0.0,
                                      std::plus<float>(),
                                      [](float a, float b){ return abs(a - b); });

            // Update global
            inputpre = pow(t, 2.0);

            // Step GeNN kernel
            stepGeNNFn();
        }

        return err;
    }
};
