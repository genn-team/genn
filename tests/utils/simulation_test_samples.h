#pragma once

// Standard C++ includes
#include <algorithm>
#include <iterator>
#include <vector>

// Standard C includes
#include <cmath>

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestSamples
//----------------------------------------------------------------------------
class SimulationTestSamples : public SimulationTest
{
public:
    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual double Test(std::vector<double> &samples) const = 0;

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        // Create vector of samples
        std::vector<double> samples;
        samples.reserve(1000 * 1000);

        // Simulate to gather samples
        for (unsigned int i = 0; i < 1000; i++) {
            // Step GeNN
            StepGeNN();

            // Copy this timestep's samples into vector
            std::copy_n(xPop, 1000, std::back_inserter(samples));
        }

        // Return the result of testing bins
        return Test(samples);
    }
};