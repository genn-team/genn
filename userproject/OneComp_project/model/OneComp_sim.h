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

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

#define DBG_SIZE 10000

//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 100.0
#define TOTAL_TME 5000

CStopWatch timer;

#include "OneComp_model.h"
#include "OneComp_model.cc"

