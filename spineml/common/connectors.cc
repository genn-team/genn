#include "connectors.h"

// Standard C++ includes
#include <iostream>

// Standard C includes
#include <cassert>
#include <cmath>

// pugixml includes
#include "pugixml/pugixml.hpp"

// GeNN includes
#include "synapseMatrixType.h"

// Anonymous namespace
namespace
{

double factln(int n)
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

double bico(int n, int k)
{
    return floor(0.5 + exp(factln(n) - factln(k) - factln(n - k)));
}
//----------------------------------------------------------------------------
//  Purpose:
//
//    binomialPDF evaluates the Binomial PDF.
//
//  Discussion:
//
//    PDF(A,B;X) is the probability of exactly X successes in A trials,
//    given that the probability of success on a single trial is B.
//
//    PDF(A,B;X) = C(N,X) * B^X * ( 1.0 - B )^( A - X )
//
//    Binomial_PDF(1,B;X) = Bernoulli_PDF(B;X).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    10 October 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int X, the desired number of successes.
//    0 <= X <= A.
//
//    Input, int A, the number of trials.
//    1 <= A.
//
//    Input, double B, the probability of success on one trial.
//    0.0 <= B <= 1.0.
//
//    Output, double binomialPDF, the value of the PDF.
//
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
        return bico(a, x) * pow(b, x) * pow(1.0 - b, a - x);
    }
}
//----------------------------------------------------------------------------
//
//  Purpose:
//
//    binomialInverseCDF inverts the Binomial CDF.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    10 October 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double CDF, the value of the CDF.
//    0.0 <= CDF <= 1.0.
//
//    Input, int A, the number of trials.
//    1 <= A.
//
//    Input, double B, the probability of success on one trial.
//    0.0 <= B <= 1.0.
//
//    Output, int binomialInverseCDF, the corresponding argument.
//
int binomialInverseCDF(double cdf, int a, double b)
{
    if(cdf < 0.0 || 1.0 < cdf) {
        throw std::runtime_error("binomialInverseCDF error - CDF < 0 or 1 < CDF");
    }

    double cdf2 = 0.0;
    for (int x = 0; x <= a; x++)
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
SynapseMatrixType getMatrixType(unsigned int numPre, unsigned int numPost, unsigned int meanRowLength, bool globalG)
{
    // Calculate the size of the dense weight matrix
    const unsigned int denseSize = numPre * numPost * sizeof(float);    //**TODO** variable weight types

    // Estimate number of sparse synapses required
    const unsigned int numSparseSynapses = numPre * meanRowLength;

    // Calculate the overheads of the Yale sparse matrix format
    const unsigned int sparseDataStructureSize = sizeof(unsigned int) * (numPre + 1 + numSparseSynapses);

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
        const unsigned int sparseSize = sparseDataStructureSize + (sizeof(float) * numSparseSynapses);
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
// SpineMLCommon::Connectors::FixedProbability
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLCommon::Connectors::FixedProbability::getMatrixType(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost, bool globalG)
{
    const double connectionProbability = node.attribute("probability").as_double();

    // If we're implementing a dense matrix and individual
    //  weights aren't required we can use DENSE_GLOBALG
    if(connectionProbability == 1.0 && globalG) {
        std::cout << "\tFully-connected FixedProbability connector implemented as DENSE_GLOBALG" << std::endl;
        return SynapseMatrixType::DENSE_GLOBALG;
    }
    else {
        const unsigned int meanRowLength = (unsigned int)((double)numPost * connectionProbability);
        return ::getMatrixType(numPre, numPost, meanRowLength, globalG);
    }
}
//----------------------------------------------------------------------------
unsigned int SpineMLCommon::Connectors::FixedProbability::estimateMaxRowLength(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost)
{
    const double connectionProbability = node.attribute("probability").as_double();

    // Calculate suitable quantile for 0.9999 change when drawing numPre times
    const double quantile = pow(0.9999, 1.0 / (double)numPre);

    unsigned int maxRowLength = binomialInverseCDF(quantile, numPost, connectionProbability);
    std::cout << "\tFixed probability:" << connectionProbability << ", num pre:" << numPre << ", num post:" << numPost << " - Max row length:" << maxRowLength << std::endl;
    return maxRowLength;
}


//----------------------------------------------------------------------------
// SpineMLCommon::Connectors::OneToOne
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLCommon::Connectors::OneToOne::getMatrixType(const pugi::xml_node&, unsigned int numPre, unsigned int numPost, bool globalG)
{
    // If we're connecting to a single postsynaptic neuron and
    // individual weights aren't required we can use DENSE_GLOBALG
    if(numPost == 1 && globalG) {
        std::cout << "\tOne-to-one connector to one neuron postsynaptic population implemented as DENSE_GLOBALG" << std::endl;
        return SynapseMatrixType::DENSE_GLOBALG;
    }
    else {
        return ::getMatrixType(numPre, numPost, 1, globalG);
    }
}
//----------------------------------------------------------------------------
unsigned int SpineMLCommon::Connectors::OneToOne::estimateMaxRowLength(const pugi::xml_node&, unsigned int, unsigned int)
{
    return 1;
}


//----------------------------------------------------------------------------
// SpineMLGenerator::Connectors::AllToAll
//----------------------------------------------------------------------------
SynapseMatrixType SpineMLCommon::Connectors::AllToAll::getMatrixType(const pugi::xml_node&, unsigned int, unsigned int, bool globalG)
{
    return globalG ? SynapseMatrixType::DENSE_GLOBALG : SynapseMatrixType::DENSE_INDIVIDUALG;
}
//----------------------------------------------------------------------------
unsigned int SpineMLCommon::Connectors::AllToAll::estimateMaxRowLength(const pugi::xml_node&, unsigned int, unsigned int numPost)
{
    return numPost;
}

//----------------------------------------------------------------------------
// SpineMLCommon::Connectors::List
//----------------------------------------------------------------------------
/*SynapseMatrixType SpineMLCommon::Connectors::List::getMatrixType(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost, bool globalG)
{

}
//----------------------------------------------------------------------------
unsigned int SpineMLCommon::Connectors::List::estimateMaxRowLength(const pugi::xml_node &node, unsigned int numPre, unsigned int numPost)
{
}*/