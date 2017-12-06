#include "connectors.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>
#include <cstdint>

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
unsigned int createFixedProbabilitySparse(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost,
                                          SparseProjection &sparseProjection, SpineMLSimulator::Connectors::AllocateFn allocateFn)
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
unsigned int createOneToOneSparse(const pugi::xml_node &, unsigned int numPre, unsigned int numPost,
                                  SparseProjection &sparseProjection, SpineMLSimulator::Connectors::AllocateFn allocateFn)
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
unsigned int createListSparse(const pugi::xml_node &node, unsigned int numPre, unsigned int,
                              SparseProjection &sparseProjection, SpineMLSimulator::Connectors::AllocateFn allocateFn,
                              const filesystem::path &basePath, std::vector<unsigned int> &remapIndices)
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

    // Create temporary vector to hold indices
    std::vector<std::pair<unsigned int, unsigned int>> tempIndices;
    tempIndices.reserve(numConnections);

    // Zero all indInG
    // **NOTE** these are initially used to store row lengths
    std::fill(&sparseProjection.indInG[0], &sparseProjection.indInG[numPre + 1], 0);
    
    // If connectivity is specified using a binary file
    if(binaryFile) {
        // Create approximately 1Mbyte buffer to hold pre and postsynaptic indices
        // **NOTE** this is also a multiple of both 2 and 3 so
        // will hold a whole number of either format of synapse
        const unsigned int bufferSize = 2 * 3 * 43690;
        std::vector<uint32_t> connectionBuffer(bufferSize);

        // If there are individual delays then each synapse is 3 words rather than 2
        const bool explicitDelay = (binaryFile.attribute("explicit_delay_flag").as_uint() != 0);
        const unsigned int wordsPerSynapse = explicitDelay ? 3 : 2;

        // Read binary connection filename from node
        std::string filename = (basePath / binaryFile.attribute("file_name").value()).str();

        // Open file for binary IO
        std::ifstream input(filename, std::ios::binary);
        if(!input.good()) {
            throw std::runtime_error("Cannot open binary connection file:" + filename);
        }

        // Loop through binary words
        for(size_t remainingWords = numConnections * wordsPerSynapse; remainingWords > 0;) {
            // Read a block into buffer
            const size_t blockWords = std::min<size_t>(bufferSize, remainingWords);
            input.read(reinterpret_cast<char*>(connectionBuffer.data()), blockWords * sizeof(uint32_t));

            // Check block was read succesfully
            if((size_t)input.gcount() != (blockWords * sizeof(uint32_t))) {
                throw std::runtime_error("Unexpected end of binary connection file");
            }

            // Loop through synapses in buffer
            for(size_t w = 0; w < blockWords; w += wordsPerSynapse) {
                // Add to temporary indices
                tempIndices.emplace_back(connectionBuffer[w], connectionBuffer[w + 1]);

                // Increment row length
                sparseProjection.indInG[connectionBuffer[w]]++;
            }

            // Subtract words in block from totalConnectors
            remainingWords -= blockWords;
        }
    }
    // Otherwise loop through connections and add to projection
    else {
        // Loop through connections
        for(auto c : connections) {
            // Add to temporary indices
            const unsigned int pre = c.attribute("src_neuron").as_uint();
            const unsigned int post = c.attribute("dst_neuron").as_uint();
            tempIndices.emplace_back(pre, post);

            // Increment row length
            sparseProjection.indInG[pre]++;
        }
    }

    // Resize remap indices and initialise to SpineML order
    remapIndices.resize(numConnections);
    std::iota(remapIndices.begin(), remapIndices.end(), 0);

    // Sort indirectly (using remap indices) so connections are in SparseProjection order
    std::sort(remapIndices.begin(), remapIndices.end(),
              [&tempIndices](unsigned int a, unsigned int b)
              {
                  return (tempIndices[a] < tempIndices[b]);
              });


    // Calculate partial sum of row lengths to build presynaptic indices
    std::partial_sum(&sparseProjection.indInG[0], &sparseProjection.indInG[numPre],
                     &sparseProjection.indInG[0]);

    // Because partial sum doesn't handle overlap - copy the partial sums forward
    // by one element and insert a zero in first index to make correct structure
    std::copy_backward(&sparseProjection.indInG[0], &sparseProjection.indInG[numPre],
                       &sparseProjection.indInG[numPre + 1]);
    sparseProjection.indInG[0] = 0;

    // Reorder temporary postsynaptic indices into sparse projection
    std::transform(remapIndices.cbegin(), remapIndices.cend(), &sparseProjection.ind[0],
                   [&tempIndices](unsigned int index)
                   {
                       return tempIndices[index].second;
                   });

    std::cout << "\tList connector with " << numConnections << " sparse synapses" << std::endl;

    // Check connection building has produced a data structure with the right number of synapses
    assert(sparseProjection.indInG[numPre] == numConnections);

    return numConnections;
}
}   // anonymous namespace

//------------------------------------------------------------------------
// SpineMLSimulator::Connectors
//------------------------------------------------------------------------
unsigned int SpineMLSimulator::Connectors::create(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost,
                                                  SparseProjection *sparseProjection, AllocateFn allocateFn,
                                                  const filesystem::path &basePath, std::vector<unsigned int> &remapIndices)
{
    auto oneToOne = node.child("OneToOneConnection");
    if(oneToOne) {
        if(sparseProjection != nullptr && allocateFn != nullptr) {
            return createOneToOneSparse(oneToOne, numPre, numPost,
                                        *sparseProjection, allocateFn);
        }
        else {
            throw std::runtime_error("OneToOneConnection does not have corresponding SparseProjection structure and allocate function");
        }
    }

    auto allToAll = node.child("AllToAllConnection");
    if(allToAll) {
        if(sparseProjection == nullptr && allocateFn == nullptr) {
            return (numPre * numPost);
        }
        else {
            throw std::runtime_error("AllToAllConnection should not have SparseProjection structure or allocate function");
        }
    }

    auto fixedProbability = node.child("FixedProbabilityConnection");
    if(fixedProbability) {
        if(sparseProjection != nullptr && allocateFn != nullptr) {
            return createFixedProbabilitySparse(fixedProbability, numPre, numPost,
                                                *sparseProjection, allocateFn);
        }
        else if(fixedProbability.attribute("probability").as_double() == 1.0) {
            return (numPre * numPost);
        }
        else {
            throw std::runtime_error("Unless connection probability is 1.0, FixedProbabilityConnection requires SparseProjection structure and allocate function");
        }
    }

    auto connectionList = node.child("ConnectionList");
    if(connectionList) {
        if(sparseProjection != nullptr && allocateFn != nullptr) {
            return createListSparse(connectionList, numPre, numPost,
                                    *sparseProjection, allocateFn, basePath,
                                    remapIndices);
        }
        else {
            throw std::runtime_error("ConnectionList does not have corresponding SparseProjection structure and allocate function");
        }
    }

    throw std::runtime_error("No supported connection type found for projection");
}