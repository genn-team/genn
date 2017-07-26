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
#include "filesystem/path.h"

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "synapseMatrixType.h"

// Anonymous namespace
namespace
{
// Calculate log factorial using lookup table and log gamma function from standard library
// Adopted from numerical recipes in C p170
double lnFact(int n)
{
    // **NOTE** a static array is automatically initialized  to zero.
    static double a[101];

    if (n < 0) {
        throw std::runtime_error("Negative factorial in routine factln");
    }
    else if (n <= 1) {
        return 0.0;
    }
    //In range of table.
    else if (n <= 100) {
        return a[n] ? a[n] : (a[n] = lgamma(n + 1.0));
    }
    // Out  of range  of  table.
    else {
        return lgamma(n + 1.0);
    }
}
//----------------------------------------------------------------------------
// Calculate binomial coefficient using log factorial
// Adopted from numerical recipes in C p169
double binomialCoefficient(int n, int k)
{
    return floor(0.5 + exp(lnFact(n) - lnFact(k) - lnFact(n - k)));
}
//----------------------------------------------------------------------------
// Evaluates PDF of binomial distribution
// Adopted from C++ 'prob' libray found https://people.sc.fsu.edu/~jburkardt/
double binomialPDF(int x, int a, double b)
{
    if(a < 1) {
        return 0.0;
    }
    else if(x < 0 || a < x) {
        return 0.0;
    }
    else if(b == 0.0) {
        if(x == 0) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else if(b == 1.0) {
        if(x == a) {
           return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else {
        return binomialCoefficient(a, x) * pow(b, x) * pow(1.0 - b, a - x);
    }
}
//----------------------------------------------------------------------------
// Evaluates inverse CDF of binomial distribution
// Adopted from C++ 'prob' libray found https://people.sc.fsu.edu/~jburkardt/
unsigned int binomialInverseCDF(double cdf, unsigned int a, double b)
{
    if(cdf < 0.0 || 1.0 < cdf) {
        throw std::runtime_error("binomialInverseCDF error - CDF < 0 or 1 < CDF");
    }

    double cdf2 = 0.0;
    for (unsigned int x = 0; x <= a; x++)
    {
        const double pdf = binomialPDF(x, a, b);
        cdf2 += pdf;

        if (cdf <= cdf2) {
            return x;
        }

    }

    throw std::runtime_error("Invalid CDF parameterse");
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
            std::cout << "\tImplementing as SPARSE_GLOBALG" << std::endl;
            return SynapseMatrixType::SPARSE_GLOBALG;
        }
        else {
            std::cout << "\tImplementing as DENSE_INDIVIDUALG" << std::endl;
            return SynapseMatrixType::DENSE_INDIVIDUALG;
        }
    }
    // Otherwise, if we have to use individual weights, compare size
    // of sparse matrix structure and weights against dense matrix
    else {
        const unsigned int sparseSize = sparseDataStructureSize + (sizeof(float) * numSynapses);
        if(sparseSize < denseSize) {
            std::cout << "\tImplementing as SPARSE_INDIVIDUALG" << std::endl;
            return SynapseMatrixType::SPARSE_INDIVIDUALG;
        }
        else {
            std::cout << "\tImplementing as DENSE_INDIVIDUALG" << std::endl;
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
        std::cout << "\tFully-connected FixedProbability connector implemented as DENSE" << std::endl;
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
    std::cout << "\tFixed probability:" << connectionProbability << ", num pre:" << numPre << ", num post:" << numPost << " - Max row length:" << maxRowLength << std::endl;
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
    float explicitDelayValue = std::numeric_limits<float>::quiet_NaN();
    bool heterogeneousWarningShown = false;
    if(binaryFile) {
        // If there are individual delays then each synapse is 3 words rather than 2
        const bool explicitDelay = (binaryFile.attribute("explicit_delay_flag").as_uint() != 0);
        const unsigned int wordsPerSynapse = explicitDelay ? 3 : 2;

        // Read number of connections and filename of file fr
        const unsigned int numConnections = binaryFile.attribute("num_connections").as_uint();
        std::string filename = (basePath / binaryFile.attribute("file_name").value()).str();

        // Open file for binary IO
        std::ifstream input(filename, std::ios::binary);
        if(!input.good()) {
            throw std::runtime_error("Cannot open binary connection file:" + filename);
        }

        // Create approximately 1Mbyte buffer to hold pre and postsynaptic indices
        // **NOTE** this is also a multiple of both 2 and 3 so
        // will hold a whole number of either format of synapse
        constexpr unsigned int  bufferSize = 2 * 3 * 43690;
        uint32_t connectionBuffer[bufferSize];

        // Loop through
        for(size_t remainingWords = numConnections * wordsPerSynapse; remainingWords > 0;) {
            // Read a block into buffer
            const size_t blockWords = std::min<size_t>(bufferSize, remainingWords);
            input.read(reinterpret_cast<char*>(&connectionBuffer[0]), blockWords * sizeof(uint32_t));

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

                    // If this is the first delay value to be read, use as explcit delay value
                    if(std::isnan(explicitDelayValue)) {
                        explicitDelayValue = *synapseDelay;
                        std::cout << "\tReading delay value of:" << explicitDelayValue << "ms from synapse" << std::endl;
                    }
                    // Otherwise if this is different than previously read value, give error
                    else if(!heterogeneousWarningShown && std::abs(explicitDelayValue - *synapseDelay) > std::numeric_limits<float>::epsilon()) {
                        std::cout << "\tWARNING: GeNN does not support heterogeneous synaptic delays - different values (";
                        std::cout << explicitDelayValue << ", " << *synapseDelay << ") encountered" << std::endl;
                        heterogeneousWarningShown = true;
                        //throw std::runtime_error("GeNN does not support heterogeneous synaptic delays - different values (" +
                        //                         std::to_string(explicitDelayValue) + ", " + std::to_string(*synapseDelay) + ") encountered");
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

            // If this synapse has a delay
            auto delay = c.attribute("delay");
            if(delay) {
                // If this is the first delay value to be read, use as explcit delay value
                const float synapseDelay = delay.as_float();
                if(std::isnan(explicitDelayValue)) {
                    explicitDelayValue = synapseDelay;
                    std::cout << "\tReading delay value of:" << explicitDelayValue << "ms from synapse" << std::endl;
                }
                // Otherwise if this is different than previously read value, give error
                else if(!heterogeneousWarningShown && std::abs(explicitDelayValue - synapseDelay) > std::numeric_limits<float>::epsilon()) {
                    std::cout << "\tWARNING: GeNN does not support heterogeneous synaptic delays - different values (";
                    std::cout << explicitDelayValue << ", " << synapseDelay << ") encountered" << std::endl;
                    heterogeneousWarningShown = true;
                    //throw std::runtime_error("GeNN does not support heterogeneous synaptic delays - different values (" +
                    //                            std::to_string(explicitDelayValue) + ", " + std::to_string(synapseDelay) + ") encountered");
                }
            }
        }
    }

    // Return size of largest histogram bin and explicit delay value
    return std::make_pair(*std::max_element(rowLengths.begin(), rowLengths.end()), explicitDelayValue);
}