#include "connectors.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

// Standard C includes
#include <cassert>
#include <cstdint>

// Filesystem includes
#include "path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
unsigned int createListSparse(const pugi::xml_node &node, unsigned int numPre, unsigned int,
                              unsigned int *rowLength, unsigned int *ind, unsigned int maxRowLength,
                              const filesystem::path &basePath, std::vector<unsigned int> &remapIndices)
{
    // Get number of connections, either from BinaryFile
    // node attribute or by counting Connection children
    auto binaryFile = node.child("BinaryFile");
    auto connections = node.children("Connection");
    const unsigned int numConnections = binaryFile ?
        binaryFile.attribute("num_connections").as_uint() :
        std::distance(connections.begin(), connections.end());

    // Zero all indInG
    // **NOTE** these are initially used to store row lengths
    std::fill_n(rowLength, numPre, 0);
    
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
                const unsigned int pre = connectionBuffer[w];
                const unsigned int post = connectionBuffer[w + 1];

                // Add postsynaptic index to ragged data structure
                ind[(pre * maxRowLength) + rowLength[pre]] = post;
                
                // Increment row length
                rowLength[pre]++;
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
            
            // Add postsynaptic index to ragged data structure
            ind[(pre * maxRowLength) + rowLength[pre]] = post;
                
            // Increment row length
            rowLength[pre]++;
        }
    }

    // Resize remap indices and initialise to SpineML order
    //remapIndices.resize(numConnections);
    //std::iota(remapIndices.begin(), remapIndices.end(), 0);

    // Sort indirectly (using remap indices) so connections are in SparseProjection order
    /*std::sort(remapIndices.begin(), remapIndices.end(),
              [&tempIndices](unsigned int a, unsigned int b)
              {
                  return (tempIndices[a] < tempIndices[b]);
              });*/

    std::cout << "\tList connector with " << numConnections << " sparse synapses" << std::endl;

    return numConnections;
}
}   // anonymous namespace

//------------------------------------------------------------------------
// SpineMLSimulator::Connectors
//------------------------------------------------------------------------
unsigned int SpineMLSimulator::Connectors::create(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost,
                                                  unsigned int **rowLength, unsigned int **ind, unsigned int *maxRowLength,
                                                  const filesystem::path &basePath, std::vector<unsigned int> &remapIndices)
{
    // One to one connectors are initialised using sparse connectivity initialisation
    auto oneToOne = node.child("OneToOneConnection");
    if(oneToOne) {
        return numPre;
    }

    // All to all connectors are initialised using sparse connectivity initialisation
    auto allToAll = node.child("AllToAllConnection");
    if(allToAll) {
        return numPre * numPost;
    }

    auto fixedProbability = node.child("FixedProbabilityConnection");
    if(fixedProbability) {
        if(maxRowLength != nullptr) {
            return numPre * (*maxRowLength);
        }
        else if(fixedProbability.attribute("probability").as_double() == 1.0) {
            return (numPre * numPost);
        }
        else {
            throw std::runtime_error("Unless connection probability is 1.0, FixedProbabilityConnection requires a maximum row length");
        }
    }

    auto connectionList = node.child("ConnectionList");
    if(connectionList) {
        if(rowLength != nullptr && ind != nullptr && maxRowLength != nullptr) {
            return createListSparse(connectionList, numPre, numPost,
                                    *rowLength, *ind, *maxRowLength, basePath, remapIndices);
        }
        else {
            throw std::runtime_error("ConnectionList does not have corresponding rowlength and ind arrays");
        }
    }

    throw std::runtime_error("No supported connection type found for projection");
}