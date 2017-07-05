#include "connectors.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// Standard C includes
#include <cassert>

// Filesystem includes
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "sparseProjection.h"

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
void addSynapseToSparseProjection(unsigned int i, unsigned int j, unsigned int numPre,
                                  SparseProjection &sparseProjection)
{
    // Get index of current end of row in sparse projection
    const unsigned int rowEndIndex = sparseProjection.indInG[i + 1];

    // Also get index of last synapse
    const unsigned int lastSynapseIndex = sparseProjection.indInG[numPre];

    // If these aren't the same (there are existing synapses after this one), shuffle up the indices
    if(rowEndIndex != lastSynapseIndex) {
        std::move_backward(&sparseProjection.ind[rowEndIndex], &sparseProjection.ind[lastSynapseIndex],
                            &sparseProjection.ind[lastSynapseIndex + 1]);
    }

    // Insert new synapse
    sparseProjection.ind[rowEndIndex] = j;

    // Increment all subsequent indices
    std::transform(&sparseProjection.indInG[i + 1], &sparseProjection.indInG[numPre], &sparseProjection.indInG[i + 1],
                   [](unsigned int index)
                   {
                       return (index + 1);
                   });
}
}   // anonymous namespace

//------------------------------------------------------------------------
// SpineMLSimulator::Connectors
//------------------------------------------------------------------------
unsigned int SpineMLSimulator::Connectors::fixedProbabilitySparse(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost,
                                                                  SparseProjection &sparseProjection, AllocateFn allocateFn)
{
    // Create RNG and seed if required
    std::mt19937 gen;
    auto seed = node.attribute("seed");
    if(seed) {
        gen.seed(seed.as_uint());
        std::cout << "\tSeed:" << seed.as_uint() << std::endl;
    }

    // Read probability and seed from connector
    double probability = node.attribute("probability").as_double();

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

    std::cout << "\tFixed probability connector with " << tempInd.size() << " sparse synapses" << std::endl;

    // Add final index
    tempIndInG[numPre] = tempInd.size();

    // Allocate SparseProjection arrays
    allocateFn(tempInd.size());

    // Copy indices
    std::copy(tempIndInG.begin(), tempIndInG.end(), &sparseProjection.indInG[0]);
    std::copy(tempInd.begin(), tempInd.end(), &sparseProjection.ind[0]);

    // Return number of connections
    return tempInd.size();
}
//------------------------------------------------------------------------
unsigned int SpineMLSimulator::Connectors::oneToOneSparse(const pugi::xml_node &, unsigned int numPre, unsigned int numPost,
                                                          SparseProjection &sparseProjection, AllocateFn allocateFn)
{
    if(numPre != numPost) {
        throw std::runtime_error("One-to-one connector can only be used between two populations of the same size");
    }

    std::cout << "\tOne-to-one connector with " << numPre << " sparse synapses" << std::endl;

    // Allocate SparseProjection arrays
    allocateFn(numPre);

    // Configure synaptic rows
    for(unsigned int i = 0; i < numPre; i++)
    {
        sparseProjection.indInG[i] = i;
        sparseProjection.ind[i] = i;
    }
    sparseProjection.indInG[numPre] = numPre;

    // Return number of connections
    return numPre;
}
//------------------------------------------------------------------------
unsigned int SpineMLSimulator::Connectors::listSparse(const pugi::xml_node &node, unsigned int numPre, unsigned int,
                                                      SparseProjection &sparseProjection, AllocateFn allocateFn, const filesystem::path &basePath)
{
    // Get number of connections, either from BinaryFile
    // node attribute or by counting Connection children
    auto binaryFile = node.child("BinaryFile");
    auto connections = node.children("Connection");
    const unsigned int numConnections = binaryFile ?
        binaryFile.attribute("num_connections").as_uint() :
        std::distance(connections.begin(), connections.end());

    // Allocate SparseProjection arrays
    allocateFn(numConnections);

     // Zero all indInG
    std::fill(&sparseProjection.indInG[0], &sparseProjection.indInG[numPre + 1], 0);

    // If connectivity is specified using a binary file
    if(binaryFile) {
        // If each synapse
        if(binaryFile.attribute("explicit_delay_flag").as_uint() != 0) {
            throw std::runtime_error("GeNN does not currently support individual delays");
        }

        // Read binary connection filename from node
        std::string filename = (basePath / binaryFile.attribute("file_name").value()).str();

        // Open file for binary IO
        std::ifstream input(filename, std::ios::binary);
        if(!input.good()) {
            throw std::runtime_error("Cannot open binary connection file:" + filename);
        }

        // Create 1Mbyte buffer to hold pre and postsynaptic indicces
        uint32_t connectionBuffer[262144];

        // Loop through binary file
        for(size_t remainingWords = numConnections * 2; remainingWords > 0;) {
            // Read a block into buffer
            const size_t blockWords = std::min<size_t>(262144, remainingWords);
            input.read(reinterpret_cast<char*>(&connectionBuffer[0]), blockWords * sizeof(uint32_t));

            // Check block was read succesfully
            if((size_t)input.gcount() != (blockWords * sizeof(uint32_t))) {
                throw std::runtime_error("Unexpected end of binary connection file");
            }

            // Loop through synapses in buffer and add to projection
            for(size_t w = 0; w < blockWords; w += 2) {
                addSynapseToSparseProjection(connectionBuffer[w], connectionBuffer[w + 1],
                                             numPre, sparseProjection);
            }

            // Subtract words in block from total
            remainingWords -= blockWords;
        }
    }
    // Otherwise loop through connections and add to projection
    else {
        // Loop through connections
        for(auto c : connections) {
            addSynapseToSparseProjection(c.attribute("src_neuron").as_uint(), c.attribute("trg_neuron").as_uint(),
                                         numPre, sparseProjection);
        }
    }

    // Check connection building has produced a data structure with the right number of synapses
    assert(sparseProjection.indInG[numPre] == numConnections);

    return numConnections;
}