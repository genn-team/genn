#pragma once

// Standard C++ includes
#include <vector>

// Standard C includes
#include <cmath>

// Test includes
#include "simulation_test.h"

//----------------------------------------------------------------------------
// SimulationTestHistogram
//----------------------------------------------------------------------------
class SimulationTestHistogram : public SimulationTest
{
public:
    SimulationTestHistogram(double min, double max, size_t numBins)
        : m_Min(min), m_BinWidth((max - min) / (double)numBins), m_NumBins(numBins)
    {}

    //----------------------------------------------------------------------------
    // Declared virtuals
    //----------------------------------------------------------------------------
    virtual bool Test(const std::vector<double> &bins) const = 0;

    //----------------------------------------------------------------------------
    // SimulationTest virtuals
    //----------------------------------------------------------------------------
    virtual void Init()
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool Simulate()
    {
        // Create vector of bins
        std::vector<double> bins(m_NumBins);

        // Simulate to gather samples
        for (unsigned int i = 0; i < 1000; i++) {
            // Step GeNN
            StepGeNN();

            // Loop through each neuron's output
            for(unsigned int j = 0; j < 1000; j++) {
                // Determine which bin it's in
                const int binIndex = (int)floor((xPop[j] - m_Min) / m_BinWidth);

                // If it's a bin we're calculating chi-squared over, add one to that bin
                if(binIndex >= 0 && binIndex < m_NumBins) {
                    bins[binIndex] += 1.0;
                }
            }
        }

        // Return the result of testing bins
        return Test(bins);
    }

protected:
    //----------------------------------------------------------------------------
    // Protected API
    //----------------------------------------------------------------------------
    double GetBinWidth() const{ return m_BinWidth; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const double m_Min;
    const double m_BinWidth;
    const size_t m_NumBins;
};