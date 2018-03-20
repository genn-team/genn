// Standard C++ includes
#include <algorithm>
#include <random>
#include <vector>

// Google test includes
#include "gtest/gtest.h"

// GeNN includes
#include "sparseUtils.h"

//------------------------------------------------------------------------
TEST(CreatePostToPreArrayTest, FixedProbability) {
    const unsigned int numPre = 1000;
    const unsigned int numPost = 1000;
    const double probability = 0.1;

    // Allocate memory for indices
    // **NOTE** RESIZE as this vector is populated by index
    std::vector<unsigned int> tempIndInG;
    tempIndInG.resize(numPre + 1);

    // Reserve a temporary vector to store indices
    std::vector<unsigned int> tempInd;
    tempInd.reserve((unsigned int)((float)(numPre * numPost) * probability));

    // Create RNG to draw probabilities
    std::mt19937 rng;
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Loop through pre neurons
    for(unsigned int i = 0; i < numPre; i++)
    {
        // Connections from this neuron start at current end of indices
        tempIndInG[i] = tempInd.size();

        // Loop through post neurons
        for(unsigned int j = 0; j < numPost; j++)
        {
            // If there should be a connection here, add one to temporary array
            if(dis(rng) < probability)
            {
                tempInd.push_back(j);
            }
        }
    }

    // Add final index
    tempIndInG[numPre] = tempInd.size();

    // Allocate memory for projection
    SparseProjection projection;
    projection.connN = tempInd.size();
    projection.indInG = new unsigned int[numPre + 1];
    projection.ind = new unsigned int[projection.connN];
    projection.preInd = NULL;
    projection.revIndInG = new unsigned int[numPost + 1];
    projection.revInd = new unsigned int[projection.connN];
    projection.remap = new unsigned int[projection.connN];

    // Copy indices
    std::copy(tempIndInG.begin(), tempIndInG.end(), &projection.indInG[0]);
    std::copy(tempInd.begin(), tempInd.end(), &projection.ind[0]);

    // Reverse!
    createPosttoPreArray(numPre, numPost, &projection);

    for(unsigned int j = 0; j < numPost; j++) {
        for(unsigned int s = projection.revIndInG[j]; s < projection.revIndInG[j + 1]; s++) {
            const unsigned int preIndex = projection.revInd[s];
            const unsigned int forwardRowStartIndex = projection.indInG[preIndex];
            const unsigned int forwardRowEndIndex = projection.indInG[preIndex + 1];

            auto postSynapse = std::find(&projection.ind[forwardRowStartIndex], &projection.ind[forwardRowEndIndex], j);
            EXPECT_TRUE(postSynapse != &projection.ind[forwardRowEndIndex]);

            // Check remapping makes sense
            const unsigned int remapIndex = projection.remap[s];
            EXPECT_GE(remapIndex, forwardRowStartIndex);
            EXPECT_LT(remapIndex, forwardRowEndIndex);
        }
    }

    // Free memory
    delete [] projection.remap;
    delete [] projection.revInd;
    delete [] projection.revIndInG;
    delete [] projection.ind;
    delete [] projection.indInG;
}