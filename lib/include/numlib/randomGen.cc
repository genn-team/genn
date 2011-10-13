/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-02-01
  
--------------------------------------------------------------------------*/

#ifndef RANDOMGEN_CC
#define RANDOMGEN_CC

#include "randomGen.h"

randomGen::randomGen()
{
  time_t t0= time(NULL);
  assert (t0 != -1);

  ulong seed1= t0;
  ulong seed2= t0%131313;
  ulong seed3= t0%171717;
  TheGen.srand(seed1, seed2, seed3);
}

randomGen::randomGen(ulong seed1, ulong seed2, ulong seed3)
{
  TheGen.srand(seed1, seed2, seed3);
}

double randomGen::n()
{
  a= TheGen.rand();
  a/= ULONG_MAX;
  //  cerr << a << endl;
  return a;
}

//ulong randomGen::nlong()
//{
//  return TheGen.rand();
//}

stdRG::stdRG()
{
  time_t t0= time(NULL);
  srand(t0);
  themax= RAND_MAX+1.0;
}

stdRG::stdRG(unsigned int seed)
{
  srand(seed);
  themax= RAND_MAX+1.0;
}




double stdRG::n()
{
  return rand()/themax;
}

#endif
