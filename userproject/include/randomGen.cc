/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-02-01
  
--------------------------------------------------------------------------*/

#ifndef RANDOMGEN_CC
#define RANDOMGEN_CC //!< macro for avoiding multiple inclusion during compilation

#include "randomGen.h"

//-----------------------------------------------------------------------
/*! \file randomGen.cc
\brief Contains the implementation of the ISAAC random number generator class for uniformly distributed random numbers and for a standard random number generator based on the C function rand().
*/
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
/*! \brief Constructor for the ISAAC random number generator class without giving explicit seeds.

The seeds for random number generation are generated from the internal clock of the computer during execution.
 */
//-----------------------------------------------------------------------

randomGen::randomGen()
{
  time_t t0= time(NULL);
  assert (t0 != -1);

  unsigned long seed1= t0;
  unsigned long seed2= t0%131313;
  unsigned long seed3= t0%171717;
  TheGen.srand(seed1, seed2, seed3);
}

//-----------------------------------------------------------------------
/*! \brief Constructor for the Gaussian random number generator class when seeds are provided explicitly.

The seeds are three arbitrary unsigned long integers.
 */
//-----------------------------------------------------------------------

randomGen::randomGen(unsigned long seed1, unsigned long seed2, unsigned long seed3)
{
  TheGen.srand(seed1, seed2, seed3);
}

//-----------------------------------------------------------------------
/*! \brief Function for generating a pseudo random number from a uniform distribution on the interval [0,1]
 */
//-----------------------------------------------------------------------

double randomGen::n()
{
  a= TheGen.rand();
  a/= ULONG_MAX;
  //  cerr << a << endl;
  return a;
}

//unsigned long randomGen::nlong()
//{
//  return TheGen.rand();
//}

//-----------------------------------------------------------------------
/*! \brief Function for seeding with fixed seeds.
 */
//-----------------------------------------------------------------------

void randomGen::srand(unsigned long seed1, unsigned long seed2, unsigned long seed3)
{
  TheGen.srand(seed1, seed2, seed3);
}

//unsigned long randomGen::nlong()
//{
//  return TheGen.rand();
//}


//-----------------------------------------------------------------------
/*! \brief Constructor of the standard random number generator class without explicit seed

The seed is taken from teh internal clock of the computer.
*/
//-----------------------------------------------------------------------

stdRG::stdRG()
{
  time_t t0= time(NULL);
  srand(t0);
  themax= RAND_MAX+1.0;
}

//-----------------------------------------------------------------------
/*! \brief Constructor of the standard random number generator class with explicit seed

The seed is an arbitrary unsigned int
*/
//-----------------------------------------------------------------------

stdRG::stdRG(unsigned int seed)
{
  srand(seed);
  themax= RAND_MAX+1.0;
}

//-----------------------------------------------------------------------
/*! \brief Method to generate a uniform random number

The moethod is a wrapper for the C function rand() and returns a pseudo random number in the interval [0,1[
*/
//----------------------------------------------------------------------- 

double stdRG::n()
{
  return rand()/themax;
}

#endif
