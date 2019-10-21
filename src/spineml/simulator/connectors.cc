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

// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
void createListSparse(const pugi::xml_node &node, double dt, unsigned int numPre, unsigned int,
                      unsigned int *rowLength, unsigned int *ind, uint8_t **delay, const unsigned int maxRowLength,
                      const filesystem::path &basePath, std::vector<unsigned int> &remapIndices)
{
    // Get number of connections, either from BinaryFile
    // node attribute or by counting Connection children
    auto binaryFile = node.child("BinaryFile");
    auto connections = node.children("Connection");
    const unsigned int numConnections = binaryFile ?
        binaryFile.attribute("num_connections").as_uint() :
        std::distance(connections.begin(), connections.end());

    // Zero row lengths
    std::fill_n(rowLength, numPre, 0);
    
    // Create array with matching dimensions to ind, initially filled with invalid value
    std::vector<unsigned int> originalOrder(numPre * maxRowLength,
                                            std::numeric_limits<unsigned int>::max());

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

        // If this connection has explict delays and no delay array was found, error
        if(explicitDelay && delay == nullptr) {
            throw std::runtime_error("Cannot build list connector with explicit delay - delay variable not found");
        }
        
        // Read binary connection filename from node
        std::string filename = (basePath / binaryFile.attribute("file_name").value()).str();

        // Open file for binary IO
        std::ifstream input(filename, std::ios::binary);
        if(!input.good()) {
            throw std::runtime_error("Cannot open binary connection file:" + filename);
        }

        // Loop through binary words
        unsigned int i = 0;
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

                // Add postsynaptic index to ragged data structure and record creation order
                const size_t index = (pre * maxRowLength) + rowLength[pre];
                ind[index] = post;
                originalOrder[index] = i++;

                // If this file contains explicit delays
                if(explicitDelay) {
                    // Cast delay word to float
                    const float *synDelay = reinterpret_cast<float*>(&connectionBuffer[w + 2]);
                    
                    // Store in delay array
                    (*delay)[index] = (uint8_t)std::round(*synDelay / dt);
                }

                // Increment row length
                rowLength[pre]++;
                assert(rowLength[pre] <= maxRowLength);
            }

            // Subtract words in block from totalConnectors
            remainingWords -= blockWords;
        }
    }
    // Otherwise loop through connections and add to projection
    else {
        // Loop through connections
        unsigned int i = 0;
        for(auto c : connections) {
            // Extract pre and postsynaptic index
            const unsigned int pre = c.attribute("src_neuron").as_uint();
            const unsigned int post = c.attribute("dst_neuron").as_uint();

            // Add postsynaptic index to ragged data structure and record creation order
            const size_t index = (pre * maxRowLength) + rowLength[pre];
            ind[index] = post;
            originalOrder[index] = i++;
            
            // If this synapse has a delay
            auto delayAttr = c.attribute("delay");
            if(delayAttr) {
                // However, if no delay array was found, error
                if(delay == nullptr) {
                    throw std::runtime_error("Cannot build list connector with explicit delay - delay variable not found");
                }
                
                // Store in delay array
                (*delay)[index] = (uint8_t)std::round(delayAttr.as_float() / dt);
            }
            
            // Increment row length
            rowLength[pre]++;
            assert(rowLength[pre] <= maxRowLength);
        }
    }

    LOGD << "\tList connector with " << numConnections << " sparse synapses";

    // Reserve remap indices array to match number of connections
    remapIndices.resize(numConnections);

    // Create array of row indices to use for sorting each row
    std::vector<unsigned int> rowOrder(maxRowLength);
    std::vector<unsigned int> rowIndCopy(maxRowLength);
    std::vector<uint8_t> rowDelayCopy(maxRowLength);

    // Loop through rows
    for(unsigned int i = 0; i < numPre; i++) {
        // Get pointer to start of row indices
        unsigned int *rowIndBegin = &ind[i * maxRowLength];

        // Copy row indices into vector
        // **NOTE** reordering in place is non-trivial
        std::copy_n(rowIndBegin, rowLength[i], rowIndCopy.begin());

        // Get iterator to end of section of row order to use for this row
        auto rowOrderEnd = rowOrder.begin();
        std::advance(rowOrderEnd, rowLength[i]);

        // Fill section with 0, 1, ..., N
        std::iota(rowOrder.begin(), rowOrderEnd, 0);

        // Sort row order based on postsynaptic indices
        std::sort(rowOrder.begin(), rowOrderEnd,
                  [&rowIndCopy](unsigned int a, unsigned int b)
                  {
                      return (rowIndCopy[a] < rowIndCopy[b]);
                  });

        // Use row order to re-order row indices back into original data structure
        std::transform(rowOrder.cbegin(), rowOrder.cend(), rowIndBegin,
                       [&rowIndCopy](unsigned int ord){ return rowIndCopy[ord]; });

        // If a delay array is present
        if(delay) {
            // Get pointer to start of row delays
            uint8_t *rowDelayBegin = &(*delay)[i * maxRowLength];

            // Copy row indices into vector
            // **NOTE** reordering in place is non-trivial
            std::copy_n(rowDelayBegin, rowLength[i], rowDelayCopy.begin());
            
            // Use row order to re-order row delays back into original data structure
            std::transform(rowOrder.cbegin(), rowOrder.cend(), rowDelayBegin,
                           [&rowDelayCopy](unsigned int ord){ return rowDelayCopy[ord]; });
        }
        
        // Loop through synapses in newly reorderd row and set the remap index in the 
        // synapse's ORIGINAL location to its new index in the ragged array
        for(unsigned int j = 0; j < rowLength[i]; j++) {
            remapIndices[originalOrder[(i * maxRowLength) + rowOrder[j]]] = (i * maxRowLength) + j;
        }
    }
}
}   // anonymous namespace

//------------------------------------------------------------------------
// SpineMLSimulator::Connectors
//------------------------------------------------------------------------
unsigned int SpineMLSimulator::Connectors::create(const pugi::xml_node &node, double dt, unsigned int numPre, unsigned int numPost,
                                                  unsigned int **rowLength, unsigned int **ind, uint8_t **delay, const unsigned int *maxRowLength,
                                                  const filesystem::path &basePath, std::vector<unsigned int> &remapIndices)
{
    // One to one connectors are initialised using sparse connectivity initialisation
    auto oneToOne = node.child("OneToOneConnection");
    if(oneToOne) {
        if(delay != nullptr) {
            throw std::runtime_error("OneToOneConnection does not support heterogeneous delays");
        }
        
        if(rowLength != nullptr && ind != nullptr && maxRowLength != nullptr) {
            return numPre;
        }
        else {
            throw std::runtime_error("OneToOneConnection does not have corresponding rowlength and ind arrays");
        }
    }

    // All to all connectors are initialised using sparse connectivity initialisation
    auto allToAll = node.child("AllToAllConnection");
    if(allToAll) {
        if(delay != nullptr) {
            throw std::runtime_error("AllToAllConnection does not support heterogeneous delays");
        }
        
        if(rowLength == nullptr && ind == nullptr && maxRowLength == nullptr) {
            return numPre * numPost;
        }
        else {
            throw std::runtime_error("AllToAllConnection should be implemented as dense connectivity without rowLength and ind arrays");
        }

    }

    auto fixedProbability = node.child("FixedProbabilityConnection");
    if(fixedProbability) {
        if(delay != nullptr) {
            throw std::runtime_error("FixedProbabilityConnection does not support heterogeneous delays");
        }
        
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
            createListSparse(connectionList, dt, numPre, numPost,
                             *rowLength, *ind, delay, *maxRowLength, basePath, remapIndices);

            return numPre * (*maxRowLength);
        }
        else {
            throw std::runtime_error("ConnectionList does not have corresponding rowlength and ind arrays");
        }
    }

    throw std::runtime_error("No supported connection type found for projection");
}
