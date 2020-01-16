#pragma once

// Standard C++ includes
#include <limits>
#include <random>

// Standard C includes
#include <cassert>

inline void preCalcRowLengths(unsigned int numPre, unsigned int numPost, size_t numConnections,
                              uint16_t *subRowLengths, std::mt19937 &rng, unsigned int numThreadsPerSpike = 1)
{
    assert(numThreadsPerSpike > 0);

    // Calculate row lengths
    const size_t numPostPerThread = (numPost + numThreadsPerSpike - 1) / numThreadsPerSpike;
    const size_t leftOverNeurons = numPost % numPostPerThread;

    size_t remainingConnections = numConnections;
    size_t matrixSize = (size_t)numPre * (size_t)numPost;

    // Loop through rows
    for(size_t i = 0; i < numPre; i++) {
        const bool lastPre = (i == (numPre - 1));

        // Loop through subrows
        for(size_t j = 0; j < numThreadsPerSpike; j++) {
            const bool lastSubRow = (j == (numThreadsPerSpike - 1));

            // If this isn't the last sub-row of the matrix
            if(!lastPre || ! lastSubRow) {
                // Get length of this subrow
                const unsigned int numSubRowNeurons = (leftOverNeurons != 0 && lastSubRow) ? leftOverNeurons : numPostPerThread;

                // Calculate probability
                const double probability = (double)numSubRowNeurons / (double)matrixSize;

                // Create distribution to sample row length
                std::binomial_distribution<size_t> rowLengthDist(remainingConnections, probability);

                // Sample row length;
                const size_t subRowLength = rowLengthDist(rng);

                // Update counters
                remainingConnections -= subRowLength;
                matrixSize -= numSubRowNeurons;

                // Add row length to array
                assert(subRowLength < std::numeric_limits<uint16_t>::max());
                *subRowLengths++ = (uint16_t)subRowLength;
            }
        }
    }

    // Insert remaining connections into last sub-row
    *subRowLengths = (uint16_t)remainingConnections;
}
