/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

using namespace std;

#include <cassert>
#include "hr_time.cpp"

#include "utils.h" // for CHECK_CUDA_ERRORS
#include <cuda_runtime.h>

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

#define DBG_SIZE 10000

// and some global variables
float t= 0.0f;
unsigned int iT= 0;

//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 1000.0
#define TOTAL_TME 500000

CStopWatch timer;

#include "OneComp_model.h"
#include "OneComp_model.cc"

