#include "binomial.h"

// Standard C++ includes
#include <stdexcept>

// Standard C includes
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

// Anonymous namespace
namespace
{
// Evaluates continued fraction for incomplete beta function by modified Lentz's method
// Adopted from numerical recipes in C p227
double betacf(double a, double b, double x)
{
    const int maxIterations = 200;
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
        const double m2 = 2.0 * m;
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
    // Otherwise, use continued fraction after making the 
    // symmetry transformation.
    else {
        return 1.0 - (bt * betacf(b, a, 1.0 - x) / b);
    }
}
}   // Anonymous namespace

// Evaluates inverse CDF of binomial distribution
unsigned int binomialInverseCDF(double cdf, unsigned int n, double p)
{
    if(cdf < 0.0 || 1.0 < cdf) {
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


// evaluates inverse CDF of binomial distribution directly from definition
unsigned int directBinomialInverseCDF(double cdf, unsigned int n, double p)
{
    if(cdf < 0.0 || 1.0 < cdf) {
        throw std::runtime_error("binomialInverseCDF error - CDF < 0 or 1 < CDF");
    }
    long double bp = n*log(1.0-p);
    long double pfac= log(p)-log(1.0-p);
    long double ptot= expl(bp);
    // Loop through ks <= n 
    for (unsigned int k = 0; k < n; k++) {
      if (ptot >= cdf) return k;
      bp+= log(n-k)-log(k+1)+pfac;
      ptot+= expl(bp);
      //      std::cerr << bp << std::endl;
      //      std::cerr << ptot << std::endl;
    }
    return n;
}
   
