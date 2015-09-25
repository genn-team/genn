/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file classol_sim.h

\brief Header file containing global variables and macros used in running the classol / MBody_individualID model.
*/
//--------------------------------------------------------------------------

#include <cassert>
using namespace std;

#include "hr_time.cpp"
#include "utils.h" // for CHECK_CUDA_ERRORS

#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
//#include <cfloat>

#include "MBody_individualID.cc"

#define MYRAND(Y,X) Y = Y * 1103515245 +12345; X= (Y >> 16);

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

#define DBG_SIZE 10000

#define PATTERNNO 100
scalar InputBaseRate= 2e-04;
//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 10000.0
#define SYN_OUT_TME 20000.0

int patSetTime;
// reset input every 100 steps == 50ms
#define PAT_TIME 100.0

// pattern goes off at 2 steps == 1 ms
#define PATFTIME 1.5

int patFireTime;


#define TOTAL_TME 5000.0

CStopWatch timer;

#include "map_classol.cc"
