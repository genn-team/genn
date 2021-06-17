#include "binomial.h"

// Standard C++ includes
#include <limits>
#include <stdexcept>

// Standard C includes
#include <cmath>
#include <cassert>

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
//! Calculates the log of the binomial coefficient n choose k
double logBinomCoefficient(unsigned int n, unsigned int k) {
    return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1);
}

//! Calculates the log of the binomial PMF
double logPMFBinomial(unsigned int n, unsigned int k, double logP, double logOneMinusP) {
    // First of all we start with the definition of the PMF:
    //               k         n - k
    // PMF(k) = |n| p   (1 - p)
    //          |k|

    // However, various aspects of this will quickly overflow so we will move this to the log domain
    // log(PMF(k)) = log|n| + klog(p) + (n - k)log(1 - p)
    //                  |k|
    return logBinomCoefficient(n, k) + ((double)k * logP) + ((double)(n - k) * logOneMinusP);
}
}

//! Evaluates the inverse CDF of the binomial distribution directly from the definition
//! The calculation is done mostly in the log domain except for the final
//! accumulation of the probabilities
unsigned int binomialInverseCDF(double cdf, unsigned int n, double p)
{
    // Validate cdf and p parameters
    if(cdf < 0.0 || cdf > 1.0) {
        throw std::runtime_error("binomialInverseCDF error - cdf < 0 or cdf > 1");
    }
    if(p < 0.0 || p > 1.0) {
        throw std::runtime_error("binomialInverseCDF error - p < 0 or p > 1");
    }

    // Handle special cases of p
    if(p == 0.0) {
        return 0;
    }
    else if(p == 1.0) {
        return n;
    }

    // Handle special cases of cdf
    if(cdf == 0.0) {
        return 0;
    }
    else if(cdf == 1.0) {
        return n;
    }

    // Handle special case of n
    if(n == 0) {
        return 0;
    }

    // While you can calculate the CDF directly using the incomplete beta function, because we need to loop through
    // k anyway, it's more efficient to calculate the PMF iteratively for each k and sum them to get the CDF.
 
    // First of all we start with the definition of the PMF:
    //               k         n - k
    // PMF(k) = |n| p   (1 - p)
    //          |k|

    // However, various aspects of this will quickly overflow sfo we will move this to the log domain
    // (1) log(PMF(k)) = log|n| + klog(p) - klog(1 - p) + nlog(1 - p)
    //                      |k|

    // Precalculate log(P) and log(1 - p) as they are used many times
    const double logP = std::log(p);
    const double logOneMinusP = std::log(1.0 - p);

    // Then, the logarithms in the middle two terms of (1) can be pre-calculated and then added on for every k:
    const double logProbRatio = logP - logOneMinusP;
    
    // Calculate the log of the minimum value that is not flushed to 0 in doube precision
    const double logMin = std::log(std::numeric_limits<double>::min());

    // If we are below the expectation value of the CDF
    if (cdf < 0.5) {
        // Binary search between 0 and the mean of the binomial distribution 
        // to find the point where the PMF starts to rise above zero
        // **NOTE** mean is NOT the peak of the distribution as it's skewed
        unsigned int kMin = 0;
        unsigned int kMax = (unsigned int)(n * p);
        while((kMax - kMin) > 100) {
            const unsigned int mid= (kMax + kMin) / 2;
            if (logPMFBinomial(n, mid, logP, logOneMinusP) > logMin) {
                kMax = mid;
            }
            else {
                kMin = mid;
            }
        }

        // As kMin is the first point the PMF is non-zero and thus adds anything to the CDF, 
        // we can start our iterative calculation of subsequent log PMFs with the log PMF of this term:
        double logPMF = logPMFBinomial(n, kMin, logP, logOneMinusP);

        // We can can then begin our calculation of the CDF by taking the exponent of this logPMF
        double cdfTotal = std::exp(logPMF);

        // Loop upwards through ks <= n
        for (unsigned int k = kMin; k < n; k++) {
            // If we have reached the CDF value we're looking for, return k
            if(cdfTotal >= cdf) {
                return k;
            }

            // Binomial coefficients can be calculated iteratively (for k from kMin to n) with:
            // (2) |  n  | =  n - k  |n|
            //     |k + 1|   ------- |k|
            //                k + 1

            // This can, again, be moved to the log domain:
            // log|  n  | = log(n - k) - log(k + 1) + log| n |
            //    |k + 1|                                | k |

            // So we can update our log PMF for the next k by adding
            // these log terms as well as the one pre-calculated earlier
            logPMF += std::log((double)(n - k) / (double)(k + 1)) + logProbRatio;

            // Add the exponent of the updated PMF to the CDF total
            cdfTotal += std::exp(logPMF);
        }
    }
    // Otherwise, if we are above the expectation value of the CDF
    else {
        // Same approach as above but counting down from high k
        cdf = 1.0 - cdf;

        // Binary search between 0 and the mean of the binomial distribution
        // to find the point where the PMF starts to rise above zero
        // **NOTE** mean is NOT the peak of the distribution as it's skewed
        unsigned int kMin = (unsigned int)(n * p);
        unsigned int kMax = n;
        while((kMax - kMin) > 100) {
            const unsigned int mid = (kMax + kMin) / 2;
            if (logPMFBinomial(n, mid, logP, logOneMinusP) > logMin) {
                kMin = mid;
            }
            else {
                kMax = mid;
            }
        }
        
        // As kMax is the last point the PMF is non-zero and thus adds anything to the CDF, 
        // we can start our iterative calculation of preceding log PMFs with the log PMF of this term:
        double logPMF = logPMFBinomial(n, kMax, logP, logOneMinusP);
        
        // We can can then begin our calculation of the CDF by taking the exponent of this logPMF
        double cdfTotal = std::exp(logPMF);

        // Loop downwards through ks >= 0
        assert(kMax >= 1);
        for (unsigned int k = (kMax - 1); k >= 0; k--) {
            // If we have reached the CDF value we're looking for, return k
            if(cdfTotal > cdf) {
                return k + 1;
            }

            // By re-arranging (2), binomial coefficients can be calculated iteratively (for k from kMax - 1 down to 0) with:
            // | n | =  k + 1  |   n  |
            // | k |   ------- | k + 1|
            //          n - k

            // This can, again, be moved to the log domain:
            // log| n | = log(k + 1) - log(n - k) + log|   n   |
            //    | k |                                | k + 1 |

            // So we can update our log PMF for the previous k by adding
            // these log terms as well as the one pre-calculated earlier
            logPMF += std::log((double)(k + 1) / (double)(n - k)) - logProbRatio;

            // Add the exponent of the updated PMF to the CDF total
            // **NOTE** we only add the term if it's not zero to avoid our sum getting renormalised into nothing
            if (logPMF > logMin) {
                cdfTotal += std::exp(logPMF);
            }
        }
    }

    // Failing to sum anything should be impossible!
    assert(false);
}
   
