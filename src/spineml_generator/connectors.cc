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
#include "third_party/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "synapseMatrixType.h"

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::FixedProbability
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLGenerator::Connectors::FixedProbability::getMatrixType(const pugi::xml_node &node, unsigned int, unsigned int, bool globalG)
{
    const double connectionProbability = node.attribute("probability").as_double();

    // If we're implementing a fully-connected matrix use DENSE format
    if(connectionProbability == 1.0) {
        LOGD << "\tFully-connected FixedProbability connector implemented as DENSE";
        return globalG ? SynapseMatrixType::DENSE_GLOBALG : SynapseMatrixType::DENSE_INDIVIDUALG;
    }
    else {
        return globalG ? SynapseMatrixType::SPARSE_GLOBALG : SynapseMatrixType::SPARSE_INDIVIDUALG;
    }
}
//----------------------------------------------------------------------------
InitSparseConnectivitySnippet::Init SpineMLGenerator::Connectors::FixedProbability::getConnectivityInit(const pugi::xml_node &node)
{
    const double connectionProbability = node.attribute("probability").as_double();
    LOGD << "\tFixed probability:" << connectionProbability;

    return InitSparseConnectivitySnippet::Init(InitSparseConnectivitySnippet::FixedProbability::getInstance(), { connectionProbability });
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::OneToOne
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLGenerator::Connectors::OneToOne::getMatrixType(const pugi::xml_node&, unsigned int, unsigned int, bool globalG)
{
    return globalG ? SynapseMatrixType::SPARSE_GLOBALG : SynapseMatrixType::SPARSE_INDIVIDUALG;
}
//----------------------------------------------------------------------------
InitSparseConnectivitySnippet::Init SpineMLGenerator::Connectors::OneToOne::getConnectivityInit(const pugi::xml_node &)
{
    return InitSparseConnectivitySnippet::Init(InitSparseConnectivitySnippet::OneToOne::getInstance(), {});;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::AllToAll
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLGenerator::Connectors::AllToAll::getMatrixType(const pugi::xml_node&, unsigned int, unsigned int, bool globalG)
{
    return globalG ? SynapseMatrixType::DENSE_GLOBALG : SynapseMatrixType::DENSE_INDIVIDUALG;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::List
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLGenerator::Connectors::List::getMatrixType(const pugi::xml_node&, unsigned int, unsigned int, bool globalG)
{
    return globalG ? SynapseMatrixType::SPARSE_GLOBALG : SynapseMatrixType::SPARSE_INDIVIDUALG;
}
//----------------------------------------------------------------------------
std::pair<unsigned int, float> SpineMLGenerator::Connectors::List::readMaxRowLengthAndDelay(const filesystem::path &basePath, const pugi::xml_node &node,
                                                                                            unsigned int numPre, unsigned int)
{
    // If connectivity should be read from a binary file
    std::vector<unsigned int> rowLengths(numPre);
    auto binaryFile = node.child("BinaryFile");

    bool explicitDelay = false;
    float sumDelayMs = 0.0;
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

                    // Add to total
                    sumDelayMs += *synapseDelay;
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

            // If this synapse has a delay, set explicit delay flag and add delay to total
            auto delay = c.attribute("delay");
            if(delay) {
                explicitDelay = true;
                sumDelayMs += delay.as_float();
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
        // Calculate mean delay
        const float meanDelay = sumDelayMs / (float)numConnections;

        // Give warning regarding how they will actually be implemented
        LOGW << "\tGeNN doesn't support heterogenous synaptic delays - mean delay of " << meanDelay << "ms will be used for all synapses";

        // Return size of largest histogram bin and explicit delay value
        return std::make_pair(maxRowLength, meanDelay);
    }
    // Otherwise, return NaN to signal that no explicit delays are set
    else {
        return std::make_pair(maxRowLength, std::numeric_limits<float>::quiet_NaN());
    }
}
