/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-03-08
  
--------------------------------------------------------------------------*/

#include "gauss.h"

randomGauss::randomGauss()
{
  time_t t0= time(NULL);
  assert (t0 != -1);

  ulong seed1= t0;
  ulong seed2= t0%131313;
  ulong seed3= t0%171717;
  UniGen.srand(seed1, seed2, seed3);

  s= 0.449871;
  t= -0.386595;
  a= 0.19600;
  b= 0.25472;

  r1= 0.27597;
  r2= 0.27846;
}

randomGauss::randomGauss(ulong seed1, ulong seed2, ulong seed3)
{
  UniGen.srand(seed1, seed2, seed3);

  s= 0.449871;
  t= -0.386595;
  a= 0.19600;
  b= 0.25472;

  r1= 0.27597;
  r2= 0.27846;
}


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
