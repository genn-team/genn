#pragma once

// Standard C++ includes
#include <cmath>

double rationalApproximation(double t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    const double c[] = {2.515517, 0.802853, 0.010328};
    const double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2] * t + c[1])*t + c[0]) /
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

// https://www.johndcook.com/blog/normal_cdf_inverse/
double normalCDFInverse(double p)
{
    if(p <= 0.0 || p >= 1.0)
    {
        throw std::invalid_argument("Invalid input argument - 0.0 > p < 1.0");
    }

    // Original approximation applies ifp >= 0.5
    if(p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -rationalApproximation(sqrt(-2.0 * log(p)));
    }
    // Otherwise, as normal distribution is symmetrical, flip p
    else
    {
        // F^-1(p) = G^-1(1-p)
        return rationalApproximation(sqrt(-2.0 * log(1.0 - p)));
    }
}