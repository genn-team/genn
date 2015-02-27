/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-02-01
  
--------------------------------------------------------------------------*/

#ifndef RANDOMGEN_H
#define RANDOMGEN_H //!< macro for avoiding multiple inclusion during compilation
//--------------------------------------------------------------------------
/*! \file randomGen.h
\brief header file containing the class definition for a uniform random generator based on the ISAAC random number generator
*/
//--------------------------------------------------------------------------

#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include "isaac.hpp"
#include <assert.h>

//--------------------------------------------------------------------------
/*! 
\brief Class randomGen which implements the ISAAC random number generator for uniformely distributed random numbers
 
 The random number generator initializes with system timea or explicit seeds and returns a random number according to a uniform distribution on [0,1];
  making use of the ISAAC random number generator; C++ Implementation
  by Quinn Tyler Jackson of the RG invented by Bob Jenkins Jr.
*/
//--------------------------------------------------------------------------

class randomGen
{
 private:
  QTIsaac<8, unsigned long> TheGen; //!< The underlying ISAAC random number generator
  double a;
  
 public:
  explicit randomGen();
  randomGen(unsigned long, unsigned long, unsigned long);
  ~randomGen() { }
  double n();  //!< Method to obtain a random number from a uniform ditribution on [0,1] 
  void srand(unsigned long, unsigned long, unsigned long);
};

class stdRG
{
 private:
  double themax;
    
 public:
  explicit stdRG();
  stdRG(unsigned int);
  ~stdRG() { }
  double n();
  unsigned long nlong();
};

//#include "randomGen.cc"

#endif
