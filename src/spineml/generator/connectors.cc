#include "connectors.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstdint>

// Filesystem includes
#include "path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "synapseMatrixType.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::FixedProbability
//----------------------------------------------------------------------------
namespace SpineMLGenerator
{
namespace Connectors
{
SynapseMatrixConnectivity FixedProbability::getMatrixConnectivity(const pugi::xml_node &node, unsigned int, unsigned int)
{
    const double connectionProbability = node.attribute("probability").as_double();

    // If we're implementing a fully-connected matrix use DENSE format
    if(connectionProbability == 1.0) {
        LOGD << "\tFully-connected FixedProbability connector implemented as DENSE";
        return SynapseMatrixConnectivity::DENSE;
    }
    else {
        return SynapseMatrixConnectivity::SPARSE;
    }
}
//----------------------------------------------------------------------------
InitSparseConnectivitySnippet::Init FixedProbability::getConnectivityInit(const pugi::xml_node &node)
{
    const double connectionProbability = node.attribute("probability").as_double();
    LOGD << "\tFixed probability:" << connectionProbability;

    return InitSparseConnectivitySnippet::Init(InitSparseConnectivitySnippet::FixedProbability::getInstance(), { connectionProbability });
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::OneToOne
//----------------------------------------------------------------------------
SynapseMatrixConnectivity OneToOne::getMatrixConnectivity(const pugi::xml_node&, unsigned int, unsigned int)
{
    return SynapseMatrixConnectivity::SPARSE;
}
//----------------------------------------------------------------------------
InitSparseConnectivitySnippet::Init OneToOne::getConnectivityInit(const pugi::xml_node &)
{
    return InitSparseConnectivitySnippet::Init(InitSparseConnectivitySnippet::OneToOne::getInstance(), {});;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::AllToAll
//----------------------------------------------------------------------------
SynapseMatrixConnectivity AllToAll::getMatrixConnectivity(const pugi::xml_node&, unsigned int, unsigned int)
{
    return SynapseMatrixConnectivity::DENSE;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::List
//----------------------------------------------------------------------------
SynapseMatrixConnectivity List::getMatrixConnectivity(const pugi::xml_node&, unsigned int, unsigned int)
{
    return SynapseMatrixConnectivity::SPARSE;
}
//----------------------------------------------------------------------------
std::tuple<unsigned int, List::DelayType, float> List::readMaxRowLengthAndDelay(const filesystem::path &basePath, const pugi::xml_node &node,
                                                                                unsigned int numPre, unsigned int)
{
    // If connectivity should be read from a binary file
    std::vector<unsigned int> rowLengths(numPre, 0);
    auto binaryFile = node.child("BinaryFile");

    bool explicitDelay = false;
    bool heterogenousDelay = false;
    float maxDelayMs = std::numeric_limits<float>::quiet_NaN();
    unsigned int numConnections = 0;

    if(binaryFile) {
        // Create approximately 1Mbyte buffer to hold pre and postsynaptic indices
        // **NOTE** this is also a multiple of both 2 and 3 so
        // will hold a whole number of either format of synapse
        const unsigned int bufferSize = 2 * 3 * 43690;
        std::vector<uint32_t> connectionBuffer(bufferSize);

        // If there are individual delays then each synapse is 3 words rather than 2
        explicitDelay = (binaryFile.attribute("explicit_delay_flag").as_uint() != 0);
        const unsigned int wordsPerSynapse = explicitDelay ? 3 : 2;

        // Read number of connections and filename of file fr
        numConnections = binaryFile.attribute("num_connections").as_uint();
        std::string filename = (basePath / binaryFile.attribute("file_name").value()).str();

        // Open file for binary IO
        std::ifstream input(filename, std::ios::binary);
        if(!input.good()) {
            throw std::runtime_error("Cannot open binary connection file:" + filename);
        }

        // Loop through
        for(size_t remainingWords = numConnections * wordsPerSynapse; remainingWords > 0;) {
            // Read a block into buffer
            const size_t blockWords = std::min<size_t>(bufferSize, remainingWords);
            input.read(reinterpret_cast<char*>(connectionBuffer.data()), blockWords * sizeof(uint32_t));

            // Check block was read succesfully
            if((size_t)input.gcount() != (blockWords * sizeof(uint32_t))) {
                throw std::runtime_error("Unexpected end of binary connection file");
            }

            // Loop through presynaptic words in buffer
            for(size_t w = 0; w < blockWords; w += wordsPerSynapse) {
                // Update row length histogram
                rowLengths[connectionBuffer[w]]++;

                // If this file contains explicit delays
                if(explicitDelay) {
                    // Cast delay word to float
                    const float *synapseDelay = reinterpret_cast<float*>(&connectionBuffer[w + 2]);

                    // If this is our first delay, use this as initial maximum
                    if(std::isnan(maxDelayMs)) {
                        maxDelayMs = *synapseDelay;
                    }
                    // Otherwise, if this delay isn't the same as our current max
                    else if(maxDelayMs != *synapseDelay) {
                        // Delays must be heterogenous
                        heterogenousDelay = true;

                        // Update maximum delay
                        maxDelayMs = std::max(maxDelayMs, *synapseDelay);
                    }
                }
            }

            // Subtract words in block from total
            remainingWords -= blockWords;
        }
    }
    // Otherwise loop through connections
    else {
        for(auto c : node.children("Connection")) {
            // Increment histogram bin based on source neuron
            rowLengths[c.attribute("src_neuron").as_uint()]++;

            // Increment connections counter
            numConnections++;

            // If this synapse has a delay
            auto delay = c.attribute("delay");
            if(delay) {
                // Set explicit delay flag 
                explicitDelay = true;
                
                // If this is our first delay, use this as initial maximum
                if(std::isnan(maxDelayMs)) {
                    maxDelayMs = delay.as_float();
                }
                // Otherwise, if this delay isn't the same as our current max
                else if(maxDelayMs != delay.as_float()) {
                    // Delays must be heterogenous
                    heterogenousDelay = true;

                    // Update maximum delay
                    maxDelayMs = std::max(maxDelayMs, delay.as_float());
                }
            }
            // Otherwise, if previous synapses have had specific delays, error
            else if(explicitDelay) {
                throw std::runtime_error("GeNN doesn't support connection lists with partial explicit delays");
            }
        }
    }

    // Calculate max row length from histogram
    const unsigned int maxRowLength = *std::max_element(rowLengths.begin(), rowLengths.end());

    // If there are explicit delays
    if(explicitDelay) {
        if(heterogenousDelay) {
            LOGD << "Using heterogenous delay up to a maximum of " << maxDelayMs << "ms";
            return std::make_tuple(maxRowLength, DelayType::Heterogeneous, maxDelayMs);
        }
        else {
            LOGD << "Using homogeneous delay of " << maxDelayMs << "ms";
            return std::make_tuple(maxRowLength, DelayType::Homogeneous, maxDelayMs);
        }
    }
    // Otherwise, return delay type of none
    else {
        return std::make_tuple(maxRowLength, DelayType::None, 0.0f);
    }
}
}   // namespace Connectors
}   // namespace SpineMLGenerator
