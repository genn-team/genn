/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/


#include <cassert>
using namespace std;
#include "hr_time.cpp"

#include "utils.h" // for CHECK_CUDA_ERRORS
#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include "PoissonIzh.cc"

#define MYRAND(Y,X) Y = Y * 1103515245 +12345; X= (Y >> 16); 

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

scalar InputBaseRate= 2e-02;
//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 1000.0
#define SYN_OUT_TME 2000.0

#define TOTAL_TME 5000

CStopWatch timer;

#include "PoissonIzh-model.h"
#include "PoissonIzh-model.cc"
