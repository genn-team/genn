/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-00-08
  
--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------

  Random number generator for Gaussian RV with mean 0 and standard
  deviation 1. Based on the ratio of uniforms method by A.J. Kinderman
  and J.F. Monahan and improved with quadratic boundind curves by
  J.L. Leva. Taken from Algorithm 712 ACM Trans. Math. Softw. 18 p. 454.
  (the necessary uniform RV are obtained from the ISAAC random number
  generator; C++ Implementation by Quinn Tyler Jackson of the RG invented
  by Bob Jenkins Jr.)

--------------------------------------------------------------------------*/



#ifndef GAUSS_H
#define GAUSS_H



#include <cmath>
#include "randomGen.h"

class randomGauss
{
private:
  QTIsaac<8, ulong> UniGen;
  double s, t, a, b, r1, r2; 
  double u, v;
  double x, y, q;
  int done;
  
 public:
  explicit randomGauss();
  randomGauss(ulong, ulong, ulong);
  ~randomGauss() { }
  double n();
};

#include "randomGen.cc"
#include "gauss.cc"

#endif
