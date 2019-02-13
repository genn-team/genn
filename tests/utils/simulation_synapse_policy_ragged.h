#pragma once

// Standard C includes
#include <cmath>

// Standard C++ includes
#include <cassert>
#include <functional>
#include <numeric>

// Test utils includes
#include "simulation_synapse_policy_dense.h"

//----------------------------------------------------------------------------
// SimulationSynapsePolicyRagged
//----------------------------------------------------------------------------
class SimulationSynapsePolicyRagged : public SimulationSynapsePolicyDense
{
public:
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void Init()
    {
        #define SETUP_THE_C(I)                  \
        case I:                                 \
            rowLength= rowLengthsyn##I;         \
            ind= indsyn##I;                     \
            maxRowLength = maxRowLengthsyn##I;  \
            break;

        // all different delay groups get same connectivity
        for(int i = 0; i < 10; i++) {
            // **YUCK** extract correct sparse projection
            unsigned int *rowLength = nullptr;
            unsigned int *ind = nullptr;
            unsigned int maxRowLength = 0;
            switch (i) {
                SETUP_THE_C(0)
                SETUP_THE_C(1)
                SETUP_THE_C(2)
                SETUP_THE_C(3)
                SETUP_THE_C(4)
                SETUP_THE_C(5)
                SETUP_THE_C(6)
                SETUP_THE_C(7)
                SETUP_THE_C(8)
                SETUP_THE_C(9)
            };

            assert(maxRowLength == 1);

            // loop through pre-synaptic neurons
            for(int j = 0; j < 10; j++) {
                // each pre-synatic neuron gets one target neuron
                const unsigned int trg= (j + 1) % 10;
                rowLength[j]= 1;
                ind[(j * maxRowLength)]= trg;
            }
        }

        // Superclass
        SimulationSynapsePolicyDense::Init();
    }

    template<typename UpdateFn, typename StepGeNNFn>
    float Simulate(UpdateFn updateFn, StepGeNNFn stepGeNNFn)
    {
        float err = 0.0f;
        float x[10][10];
        while(t < 20.0f) {
            // for each delay
            for (int d = 0; d < 10; d++) {
                // for all pre-synaptic neurons
                for (int j = 0; j < 10; j++) {
                    float newX;
                    if(updateFn(iT, d, j, t, newX)) {
                        x[d][j] = newX;
                    }
                    else if(iT == 0) {
                        x[d][j] = 0.0f;
                    }
                }

                // Add error for this time step to total
                err += std::inner_product(&x[d][0], &x[d][10],
                                          GetTheW(d),
                                          0.0f,
                                          std::plus<float>(),
                                          [](float a, float b){ return std::fabs(a - b); });
            }

            // Step GeNN
            stepGeNNFn();
        }

        return err;
    }
};
