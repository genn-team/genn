/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-00-08
  
--------------------------------------------------------------------------*/

#ifndef GAUSS_H
#define GAUSS_H //!< macro for avoiding multiple inclusion during compilation

//-----------------------------------------------------------------------
/*!  \file gauss.h

\brief Random number generator for Gaussian random variable with mean 0 and standard
  deviation 1. 

This random number generator is based on the ratio of uniforms method by A.J. Kinderman
  and J.F. Monahan and improved with quadratic boundind curves by
  J.L. Leva. Taken from Algorithm 712 ACM Trans. Math. Softw. 18 p. 454.
  (the necessary uniform random variables are obtained from the ISAAC random number
  generator; C++ Implementation by Quinn Tyler Jackson of the RG invented
  by Bob Jenkins Jr.).
*/
//-----------------------------------------------------------------------

#include <cmath>
#include "randomGen.h"

//-----------------------------------------------------------------------
/*! 
\brief Class random Gauss encapsulates the methods for generating random neumbers with Gaussian distribution.

A random number from a Gaussian distribution of mean 0 and standard deviation 1 is obtained by calling the method randomGauss::n().
*/
//-----------------------------------------------------------------------

class randomGauss
{
private:
  QTIsaac<8, unsigned long> UniGen;  //!< the underlying ISAAC random number generator
  double s, t, a, b, r1, r2; 
  double u, v;
  double x, y, q;
  int done;
  
 public:
  explicit randomGauss();
  randomGauss(unsigned long, unsigned long, unsigned long);
  ~randomGauss() { }
  double n();  //!< Method for obtaining a random number with Gaussian distribution
  void srand(unsigned long, unsigned long, unsigned long);
};

#include "randomGen.cc"
#include "gauss.cc"

#endif
