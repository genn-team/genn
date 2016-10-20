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
#include "hr_time.h"

#include "utils.h" // for CHECK_CUDA_ERRORS
#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif

#include "Izh_sparse.cc"

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

#define DBG_SIZE 5000

//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 5000.0
#define TOTAL_TME 100000.0

CStopWatch timer;

//#include "Izh_sparse_model.h"
#include "Izh_sparse_model.cc"
