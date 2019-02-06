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

// Anonymous namespace
namespace
{
// Evaluates continued fraction for incomplete beta function by modified Lentz's method
// Adopted from numerical recipes in C p227
double betacf(double a, double b, double x)
{
    const int maxIterations = 100;
    const double epsilon = 3.0E-7;
    const double fpMin = 1.0E-30;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double  qam = a - 1.0;
    double c = 1.0;

    // First step of Lentz’s method.
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < fpMin) {
        d = fpMin;
    }
    d = 1.0 / d;
    double h = d;
    int m;
    for(m = 1; m <= maxIterations; m++) {
        const int m2 = 2 * m;
        const double aa1 = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa1 * d;

        // One step (the even one) of the recurrence.
        if(fabs(d) < fpMin)  {
            d = fpMin;
        }
        c = 1.0 + aa1 / c;
        if(fabs(c) < fpMin) {
            c=fpMin;
        }
        d = 1.0 / d;
        h *= d * c;
        const double aa2 = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;

        // Next step of the recurrence (the odd one).
        if (fabs(d) < fpMin) {
            d = fpMin;
        }
        c = 1.0 + aa2 / c;
        if (fabs(c) < fpMin)  {
            c = fpMin;
        }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;

        // Are we done?
        if (fabs(del - 1.0) < epsilon) {
            break;
        }
    }
    if (m > maxIterations) {
        throw std::runtime_error("a or b too big, or MAXIT too small in betacf");
    }
    return h;
}
//----------------------------------------------------------------------------
// Returns the incomplete beta function Ix(a, b)
// Adopted from numerical recipes in C p227
double betai(double a, double b, double x)
{
    if (x < 0.0 || x > 1.0) {
        throw std::runtime_error("Bad x in routine betai");
    }

    // Factors in front of the continued fraction.
    double bt;
    if (x == 0.0 || x == 1.0) {
        bt = 0.0;
    }
    else {
        bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x));
    }

    // Use continued fraction directly.
    if (x < ((a + 1.0) / (a + b + 2.0))) {
        return bt * betacf(a, b, x) / a;
    }
    // Otherwise use continued fraction, after making the
    // symmetry transformation.
    else {
        return 1.0 - (bt * betacf(b, a, 1.0 - x) / b);
    }
}
//----------------------------------------------------------------------------
// Evaluates inverse CDF of binomial distribution
inline unsigned int binomialInverseCDF(double cdf, unsigned int n, double p)
{
    if(cdf < 0.0 || cdf >= 1.0) {
        throw std::runtime_error("binomialInverseCDF error - CDF < 0 or 1 < CDF");
    }

    // Loop through ks <= n
    for (unsigned int k = 0; k <= n; k++)
    {
        // Use incomplete beta function to evalauate CDF, if it's greater than desired CDF value, return k
        if (betai(n - k, 1 + k, 1.0 - p) > cdf) {
            return k;
        }

    }

    throw std::runtime_error("Invalid CDF parameters");
}
//----------------------------------------------------------------------------
/*SynapseMatrixType getMatrixType(unsigned int numPre, unsigned int numPost, unsigned int numSynapses, bool globalG)
{
    // Calculate the size of the dense weight matrix
    const unsigned int denseSize = numPre * numPost * sizeof(float);

    // Calculate the overheads of the Yale sparse matrix format
    const unsigned int sparseDataStructureSize = sizeof(unsigned int) * (numPre + 1 + numSynapses);

    // If we can use global weights compare the size of the sparse matrix structure against the dense matrix
    // **NOTE** this function assumes that DENSE_GLOBALG cannot be used
    if(globalG) {
        if(sparseDataStructureSize < denseSize) {
            LOGD << "\tImplementing as SPARSE_GLOBALG";
            return SynapseMatrixType::SPARSE_GLOBALG;
        }
        else {
            LOGD << "\tImplementing as DENSE_INDIVIDUALG";
            return SynapseMatrixType::DENSE_INDIVIDUALG;
        }
    }
    // Otherwise, if we have to use individual weights, compare size
    // of sparse matrix structure and weights against dense matrix
    else {
        const unsigned int sparseSize = sparseDataStructureSize + (sizeof(float) * numSynapses);
        if(sparseSize < denseSize) {
            LOGD << "\tImplementing as SPARSE_INDIVIDUALG";
            return SynapseMatrixType::SPARSE_INDIVIDUALG;
        }
        else {
            LOGD << "\tImplementing as DENSE_INDIVIDUALG";
            return SynapseMatrixType::DENSE_INDIVIDUALG;
        }
    }
}*/
}   // anonymous namespace

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
unsigned int SpineMLGenerator::Connectors::FixedProbability::estimateMaxRowLength(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost)
{
    const double connectionProbability = node.attribute("probability").as_double();

    // Calculate suitable quantile for 0.9999 change when drawing numPre times
    const double quantile = pow(0.9999, 1.0 / (double)numPre);

    unsigned int maxRowLength = binomialInverseCDF(quantile, numPost, connectionProbability);
    LOGD << "\tFixed probability:" << connectionProbability << ", num pre:" << numPre << ", num post:" << numPost << " - Max row length:" << maxRowLength << std::endl;
    return maxRowLength;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::OneToOne
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLGenerator::Connectors::OneToOne::getMatrixType(const pugi::xml_node&, unsigned int, unsigned int, bool globalG)
{
    return globalG ? SynapseMatrixType::SPARSE_GLOBALG : SynapseMatrixType::SPARSE_INDIVIDUALG;
}
//----------------------------------------------------------------------------
unsigned int SpineMLGenerator::Connectors::OneToOne::estimateMaxRowLength(const pugi::xml_node&, unsigned int, unsigned int)
{
    return 1;
}

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::AllToAll
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLGenerator::Connectors::AllToAll::getMatrixType(const pugi::xml_node&, unsigned int, unsigned int, bool globalG)
{
    return globalG ? SynapseMatrixType::DENSE_GLOBALG : SynapseMatrixType::DENSE_INDIVIDUALG;
}
//----------------------------------------------------------------------------
unsigned int SpineMLGenerator::Connectors::AllToAll::estimateMaxRowLength(const pugi::xml_node&, unsigned int, unsigned int numPost)
{
    return numPost;
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
        LOGW << "\tGeNN doesn't support heterogenous synaptic delays - mean delay of " << meanDelay << "ms will be used for all synapses" << std::endl;

        // Return size of largest histogram bin and explicit delay value
        return std::make_pair(maxRowLength, meanDelay);
    }
    // Otherwise, return NaN to signal that no explicit delays are set
    else {
        return std::make_pair(maxRowLength, std::numeric_limits<float>::quiet_NaN());
    }
}