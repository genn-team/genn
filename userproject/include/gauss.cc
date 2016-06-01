/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-03-08
  
--------------------------------------------------------------------------*/

#ifndef GAUSS_CC
#define GAUSS_CC //!< macro for avoiding multiple inclusion during compilation

#include "gauss.h"
//-----------------------------------------------------------------------
/*! \file gauss.cc
\brief Contains the implementation of the Gaussian random number generator class randomGauss
*/
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
/*! \brief Constructor for the Gaussian random number generator class without giving explicit seeds.

The seeds for random number generation are generated from the internal clock of the computer during execution.
 */
//-----------------------------------------------------------------------

randomGauss::randomGauss()
{
  time_t t0= time(NULL);
  assert (t0 != -1);

  unsigned long seed1= t0;
  unsigned long seed2= t0%131313;
  unsigned long seed3= t0%171717;
  UniGen.srand(seed1, seed2, seed3);

  s= 0.449871;
  t= -0.386595;
  a= 0.19600;
  b= 0.25472;

  r1= 0.27597;
  r2= 0.27846;
}

//-----------------------------------------------------------------------
/*! \brief Constructor for the Gaussian random number generator class when seeds are provided explicitly.

The seeds are three arbitrary unsigned long integers.
 */
//-----------------------------------------------------------------------

randomGauss::randomGauss(unsigned long seed1, unsigned long seed2, unsigned long seed3)
{
  UniGen.srand(seed1, seed2, seed3);

  s= 0.449871;
  t= -0.386595;
  a= 0.19600;
  b= 0.25472;

  r1= 0.27597;
  r2= 0.27846;
}

//-----------------------------------------------------------------------
/*! \brief Function for generating a pseudo random number from a Gaussian distribution
 */
//-----------------------------------------------------------------------

double randomGauss::n()
{
  do {
    done= 1;
    do {u= ((double) UniGen.rand())/ULONG_MAX;} while (u == 0.0);
    v= ((double) UniGen.rand())/ULONG_MAX; 
    v= 1.7156 * (v - 0.5);
    x= u - s;
    y= fabs(v) - t;
    q= x*x + y*(a*y - b*x);
    if (q < r1) done= 1;
    else if (q > r2) done= 0;
    else if (v*v > -4.0*log(u)*u*u) done= 0;
  } while (!done);
    
  return v/u;
}

//-----------------------------------------------------------------------
/*! \brief Function for seeding with fixed seeds.
 */
//-----------------------------------------------------------------------

void randomGauss::srand(unsigned long seed1, unsigned long seed2, unsigned long seed3)
{
  UniGen.srand(seed1, seed2, seed3);
}


#endif
