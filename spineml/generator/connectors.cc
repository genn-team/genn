#include "connectors.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>

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
SynapseMatrixType getMatrixType(unsigned int numPre, unsigned int numPost, unsigned int numSynapses, bool globalG)
{
    // Calculate the size of the dense weight matrix
    const unsigned int denseSize = numPre * numPost * sizeof(float);    //**TODO** variable weight types

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
}
}   // anonymous namespace

//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::FixedProbability
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLGenerator::Connectors::FixedProbability::getMatrixType(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost, bool globalG)
{
    const double connectionProbability = node.attribute("probability").as_double();

    // If we're implementing a dense matrix and individual
    //  weights aren't required we can use DENSE_GLOBALG
    if(connectionProbability == 1.0 && globalG) {
        std::cout << "\tFully-connected FixedProbability connector implemented as DENSE_GLOBALG" << std::endl;
        return SynapseMatrixType::DENSE_GLOBALG;
    }
    else {
        const unsigned int numConnections = (unsigned int)((double)numPre * (double)numPost * connectionProbability);
        return ::getMatrixType(numPre, numPost, numConnections, globalG);
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
SynapseMatrixType SpineMLGenerator::Connectors::OneToOne::getMatrixType(const pugi::xml_node&, unsigned int numPre, unsigned int numPost, bool globalG)
{
    // If we're connecting to a single postsynaptic neuron and
    // individual weights aren't required we can use DENSE_GLOBALG
    if(numPost == 1 && globalG) {
        std::cout << "\tOne-to-one connector to one neuron postsynaptic population implemented as DENSE_GLOBALG" << std::endl;
        return SynapseMatrixType::DENSE_GLOBALG;
    }
    else {
        return ::getMatrixType(numPre, numPost, numPre, globalG);
    }
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
SynapseMatrixType SpineMLGenerator::Connectors::List::getMatrixType(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost, bool globalG)
{
    // Determine number of connections, either by reading it from binary file attribute or counting Connection children
    auto binaryFile = node.child("BinaryFile");
    auto connections = node.children("Connection");
    const unsigned int numConnections = binaryFile
        ? binaryFile.attribute("num_connections").as_uint()
        : std::distance(connections.begin(), connections.end());

    return ::getMatrixType(numPre, numPost, numConnections, globalG);
}
//----------------------------------------------------------------------------
unsigned int SpineMLGenerator::Connectors::List::estimateMaxRowLength(const filesystem::path &basePath, const pugi::xml_node &node,
                                                                      unsigned int numPre, unsigned int)
{
    // If connectivity should be read from a binary file
    std::vector<unsigned int> rowLengths(numPre);
    auto binaryFile = node.child("BinaryFile");
    if(binaryFile) {
        // If each synapse
        if(binaryFile.attribute("explicit_delay_flag").as_uint() != 0) {
            throw std::runtime_error("GeNN does not currently support individual delays");
        }

        // Read number of connections and filename of file fr
        const unsigned int numConnections = binaryFile.attribute("num_connections").as_uint();
        std::string filename = (basePath / binaryFile.attribute("file_name").value()).str();

        // Open file for binary IO
        std::ifstream input(filename, std::ios::binary);
        if(!input.good()) {
            throw std::runtime_error("Cannot open binary connection file:" + filename);
        }

        // Create 1Mbyte buffer to hold pre and postsynaptic indicces
        uint32_t connectionBuffer[262144];

        // Loop through
        for(size_t remainingWords = numConnections * 2; remainingWords > 0;) {
            // Read a block into buffer
            const size_t blockWords = std::min<size_t>(262144, remainingWords);
            input.read(reinterpret_cast<char*>(&connectionBuffer[0]), blockWords * sizeof(uint32_t));

            // Check block was read succesfully
            if((size_t)input.gcount() != (blockWords * sizeof(uint32_t))) {
                throw std::runtime_error("Unexpected end of binary connection file");
            }

            // Loop through presynaptic words in buffer and update histogram
            for(size_t w = 0; w < blockWords; w += 2) {
                rowLengths[connectionBuffer[w]]++;
            }

            // Subtract words in block from total
            remainingWords -= blockWords;
        }
    }
    // Otherwise loop through connections and increment histogram bins based on their source neuron
    else {
        for(auto c : node.children("Connection")) {
            rowLengths[c.attribute("src_neuron").as_uint()]++;
        }
    }

    // Return size of largest histogram bin
    return *std::max_element(rowLengths.begin(), rowLengths.end());
}