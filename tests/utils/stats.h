#pragma once

// Standard C++ includes
#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

// Standard C includes
#include <cmath>

//----------------------------------------------------------------------------
// Stats::gammaSeries
//----------------------------------------------------------------------------
namespace Stats
{
//!< Returns the incomplete gamma function P(a, x)
//!< evaluated by its series representation
//!< (Numerical Recipes in C p173)
double gammaSeries(double a, double x)
{
    constexpr unsigned int maxIterations = 100;
    constexpr double epsilon = 3.0E-7;

    const double gln = lgamma(a);
    if(x <= 0.0) {
        if(x < 0.0) {
            throw std::runtime_error("x < 0");
        }
        return 0.0;
    }
    else {
        double ap = a;
        double del = 1.0 / a;
        double sum = del;

        for(unsigned int n = 0; n < maxIterations; n++) {
            ap += 1.0;
            del *= (x / ap);
            sum += del;
            if(fabs(del) < (fabs(sum) * epsilon)) {
                return sum * exp(-x + a * log(x) - gln);
            }
        }
        throw std::runtime_error("a too large or maxIterations too small");
    }
}

//----------------------------------------------------------------------------
// Stats::gammaContinuedFraction
//----------------------------------------------------------------------------
//!< Returns the incomplete gamma function Q(a, x)
//!< evaluated by its continuous fraction representation
//!< (Numerical Recipes in C p174)
double gammaContinuedFraction(double a, double x)
{
    constexpr unsigned int maxIterations = 100;
    constexpr double epsilon = 3.0E-7;

    const double gln = lgamma(a);

    double a0 = 1.0;
    double a1 = x;
    double b0 = 0.0;
    double b1 = 1.0;

    double gold = 0.0;

    double fac = 1.0;

    for(unsigned int n = 0; n < maxIterations; n++) {
        const double an = (double)(n + 1);
        const double ana = an - a;

        // One step of recurrence
        a0 = (a1 + a0 * ana) * fac;
        b0 = (b1 + b0 * ana) * fac;

        const double anf = an * fac;

        // Next step of recurrence
        a1 = x * a0 + anf * a1;
        b1 = x * b0 + anf * b1;

        // Shall we re-normalise?
        if(a1) {
            fac = 1.0 / a1;

            // New value of answer
            const double g = b1 * fac;

            // Have we converged
            if(fabs((g - gold) / g) < epsilon) {
                return exp(-x + a * log(x) - gln) * g;
            }

            gold = g;
        }
    }

    throw std::runtime_error("a too large or maxIterations too small");
}

//----------------------------------------------------------------------------
// Stats::gammaP
//----------------------------------------------------------------------------
//!< Returns the (lower) incomplete gamma function P(a,x)
//!< (Numerical Recipes in C p172)
double gammaP(double a, double x)
{
    if(x < 0.0 || a <= 0.0) {
        throw std::runtime_error("Invalid arguments");
    }

    if(x < (a + 1.0)) {
        return gammaSeries(a, x);
    }
    else {
        return 1.0 - gammaContinuedFraction(a, x);
    }
}

//----------------------------------------------------------------------------
// Stats::gammaQ
//----------------------------------------------------------------------------
//!< Returns the (upper) incomplete gamma function Q(a,x) = 1 - P(a,x)
//!< (Numerical Recipes in C p173)
double gammaQ(double a, double x)
{
    if(x < 0.0 || a <= 0.0) {
        throw std::runtime_error("Invalid arguments");
    }

    if(x < (a + 1.0)) {
        return (1.0 - gammaSeries(a, x));
    }
    else {
        return gammaContinuedFraction(a, x);
    }
}

//----------------------------------------------------------------------------
// Stats::probks
//----------------------------------------------------------------------------
double probks(double alam)
{
    constexpr double eps1 = 0.001;
    constexpr double eps2 = 1.0e-8;

    double fac = 2.0;
    double sum = 0.0;
    double termbf = 0.0;
    const double a2 = -2.0 * alam * alam;

    for(unsigned int j = 1; j <= 100; j++) {
        const double term = fac * exp(a2 * (double)j * (double)j);
        sum += term;
        if(fabs(term) <= (eps1 * termbf) || fabs(term) <= (eps2 * sum)) {
            return sum;
        }
        fac = -fac;
        termbf = fabs(term);
    }

    // Failing to converse
    return 1.0;
}

//----------------------------------------------------------------------------
// Stats::chiSquared
//----------------------------------------------------------------------------
//!< Given the array bins, containing the observed number of events and an array ebins containing
//!< the expected number of events, and given the number of constraints (normally zero), this routine
//!< returns  the number of degrees of freedom, the chi-squared and the significance prob
//!< (Numerical Recipes in C p489)
std::tuple<double, double, double> chiSquaredTest(const std::vector<double> &bins, const std::vector<double> &ebins, int knstrn = 0)
{
    if(bins.size() != ebins.size()) {
        throw std::runtime_error("Bin count mismatch");
    }

    const double df = (double)(bins.size() - 1 - knstrn);
    double chsq = 0.0;

    for(unsigned int j = 0; j < bins.size(); j++) {
        if(ebins[j] <= 0.0) {
            throw std::runtime_error("Bad expected number");
        }

        const double temp = bins[j] - ebins[j];
        chsq += temp * temp / ebins[j];
    }

    const double prob = gammaQ(0.5 * df, 0.5 * chsq);
    return std::make_tuple(df, chsq, prob);
}

//----------------------------------------------------------------------------
// Stats::uniformCDF
//----------------------------------------------------------------------------
//!< Cumulative distribution function for standardised
//!< uniform distribution for use in tests
double uniformCDF(double x)
{
    if(x < 0.0) {
        return 0.0;
    }
    else if(x >= 1.0) {
        return 1.0;
    }
    else {
        return x;
    }
}

//----------------------------------------------------------------------------
// Stats::normalCDF
//----------------------------------------------------------------------------
//!< Cumulative distribution function for standardised
//!< normal distribution for use in tests
double normalCDF(double x)
{
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

//----------------------------------------------------------------------------
// Stats::exponentialCDF
//----------------------------------------------------------------------------
//!< Cumulative distribution function for standardised
//!< exponential distribution for use in tests
double exponentialCDF(double x)
{
    return (1.0 - exp(-x));
}

//----------------------------------------------------------------------------
// Stats::gammaCDF
//----------------------------------------------------------------------------
//!< Cumulative distribution function for
//!< gamma distribution for use in tests
double gammaCDF(double a, double x)
{
    return gammaP(a, x);
}

//----------------------------------------------------------------------------
// Stats::kolmogorovSmirnovTest
//----------------------------------------------------------------------------
//!< Given the array data and given a user-supplied callable of the single
//!< Variable cdf which is a cumulative distribution function ranging from 0
//!< (for the smallest value of its argument) to 1 (for the largest value of its
//!< argument), this routine returns the K-S statistic and the significance level.
//!< (Numerical Recipes in C p492)
template<typename F>
std::tuple<double, double> kolmogorovSmirnovTest(std::vector<double> &data, F cdf)
{
    // Sort data
    std::sort(data.begin(), data.end());

    const double en = (double)data.size();
    double d = 0.0;
    double fo = 0.0;

    // Loop over sorted cdf
    for(size_t j = 0; j < data.size(); j++) {
        // Data's CDF after this step
        const double fn = (double)(j + 1) / en;

        // Compare to use-supplied function
        const double ff = cdf(data[j]);
        const double dt = std::max(fabs(fo - ff), fabs(fn - ff));

        // Maxium distance
        if(dt > d) {
            d = dt;
        }
        fo = fn;
    }

    // Compute significance and return
    return std::make_tuple(d, probks(sqrt(en) * d));
}
}  // namespace Stats
