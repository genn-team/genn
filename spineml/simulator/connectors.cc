#include "connectors.h"

// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "sparseProjection.h"

//------------------------------------------------------------------------
// SpineMLSimulator::Connectors
//------------------------------------------------------------------------
unsigned int SpineMLSimulator::Connectors::fixedProbabilitySparse(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost,
                                                                  SparseProjection &sparseProjection, AllocateFn allocateFn)
{
    // Read probability and seed from connector
    double probability = node.attribute("probability").as_double();
    unsigned int seed = node.attribute("seed").as_uint();

    // Create RNG
    std::mt19937 gen(seed);

    // Allocate memory for indices
    // **NOTE** RESIZE as this vector is populated by index
    std::vector<unsigned int> tempIndInG;
    tempIndInG.resize(numPre + 1);

    // Reserve a temporary vector to store indices
    std::vector<unsigned int> tempInd;
    tempInd.reserve((unsigned int)((double)(numPre * numPost) * probability));

    // Create RNG to draw probabilities
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
            if(dis(gen) < probability)
            {
                tempInd.push_back(j);
            }
        }
    }

    std::cout << "\t\t" << "Fixed probability connector with " << tempInd.size() << " sparse synapses" << std::endl;

    // Add final index
    tempIndInG[numPre] = tempInd.size();

    // Allocate SparseProjection arrays
    // **NOTE** shouldn't do directly as underneath it may use CUDA or host functions
    allocateFn(tempInd.size());

    // Copy indices
    std::copy(tempIndInG.begin(), tempIndInG.end(), &sparseProjection.indInG[0]);
    std::copy(tempInd.begin(), tempInd.end(), &sparseProjection.ind[0]);

    // Return number of connections
    return tempInd.size();
}