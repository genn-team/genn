#include "binomial.h"

// Standard C++ includes
#include <stdexcept>

// Standard C includes
#include <cmath>
#include <limits>
#include <cassert>
#include <iostream>

double logChoose(int n, int k) {
    return std::lgamma(double(n+1)) - std::lgamma(double(k+1)) - std::lgamma(double(n-k+1));
}
double logPMFBinomial(double p, int n, int k) {
  return logChoose(n, k) + (double)k*std::log(p) + (double)(n-k)*std::log(1-p);
}

// Evaluates the inverse CDF of the binomial distribution directly from the definition
// The calculation is done mostly in the log domain except for the final
// accumulation of the probabilities
unsigned int binomialInverseCDF(double cdf, unsigned int n, double p)
{
    if(cdf < 0.0 || 1.0 < cdf) {
        throw std::runtime_error("binomialInverseCDF error - CDF < 0 or 1 < CDF");
    }

    // While you can calculate the CDF directly using the incomplete beta function, because we need to loop through
    // k anyway, it's more efficient to calculate the PMF iteratively for each k and sum them to get the CDF.
 
    // First of all we start with the definition of the PMF:
    //                   k         n - k
    // (1) PMF(k) = |n| p   (1 - p)
    //              |k|

    // However, various aspects of this will quickly overflow so we will move this to the log domain
    // (2) log(PMF(k)) = log|n| + klog(p) - klog(1 - p) + nlog(1 - p)
    //                      |k|

    // The logarithms in the middle two terms can be pre-calculated and then added on for every k:
    const double logProbRatio = log((double)p) - log((double)(1.0 - p));

    if (cdf < 0.5) {
        // The final term is a constant so we can again calculate it once at the start of our sum:
        double logPMF = (double)n * log((double)(1.0 - p));

        // Because the first three terms of (2) will be zero for k=0,
        // we can can calculate the CDF by taking the exponent of the constant term
        double cdfTotal = exp(logPMF);

        // Loop through ks <= n
        for (unsigned int k = 0; k < n; k++) {
            // If we have reached the CDF value we're looking for, return k
            if(cdfTotal >= cdf) {
                return k;
            }

            // Binomial coefficients can be calculated iteratively (for k from 0 to n) with:
            // |  n  | =  n - k  |n|
            // |k + 1|   ------- |k|
            //            k + 1

            // This can, again, be moved to the log domain:
            // log|  n  | = log(n - k) - log(k + 1) + log| n |
            //    |k + 1|                                | k |

            // So we can update our log PMF for the next k by adding
            // these log terms as well as the one pre-calculated earlier
            logPMF += log((double)(n - k)) - log((double)(k + 1)) + logProbRatio;

            // Add the exponent of the updated PMF to the CDF total
            cdfTotal += exp(logPMF);
        }
        return n;
    }
    else {
        // Same approach as above but counting down from high k
        cdf= 1.0 - cdf;
	// log of the minimum value that is not flushed to 0 in long doube precision
	double logMin = logl(std::numeric_limits<double>::min());
	// Expectation value and standard deviation
	double mu= n*p;
	double sig= sqrt(n*p*(1-p));
	// Guesstimate when the terms of logPMF are getting large enough so that exp(logPMF) is not flushed to 0
	// based on the normal approximation of the binomial distribution.
	// Adding sigma as a margin for error of the used normal approximation
	unsigned int kmax= (unsigned int) (mu+sig*sqrt(-2.0*(log(sig*sqrt(2*M_PI))+logMin))+sig);
	kmax= std::min(kmax, n);
	// Initialize with the logPMF value at kmax
	double logPMF= logPMFBinomial(p, n, kmax);
	std::cout << logPMF << " " << logMin << std::endl;
	assert(logPMF <= logMin);
	// add the initial term to cdfTotal (likely still flushed to 0 at this point)
        double cdfTotal = exp(logPMF);
	
        // Loop through ks >= 0
        for (unsigned int k = kmax-1; k >= 0; k--) {
            // If we have reached the CDF value we're looking for, return k
            if(cdfTotal > cdf) {
                return k + 1;
            }

            // Binomiral coefficients can be calculated iteratively (for k from n - 1 down to 0) with:
            // | n | =  k + 1  |   n  |
            // | k |   ------- | k + 1|
            //          n - k

            // This can, again, be moved to the log domain:
            // log| n | = log(k + 1) - log(n - k) + log|   n   |
            //    | k |                                | k + 1 |

            // So we can update our log PMF for the previous k by adding
            // these log terms as well as the one pre-calculated earlier
            logPMF += log((double)(k + 1)/(double)(n - k)) - logProbRatio;

            // Add the exponent of the updated PMF to the CDF total
	    if (logPMF > logMin) {
	        cdfTotal += exp(logPMF);
	    }
        }
        return 0;
    }
}
   
