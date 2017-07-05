/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file userproject/MBody1_project/model/classol_sim.h

\brief Header file containing global variables and macros used in running the classol / MBody1 model.
*/
//--------------------------------------------------------------------------

#include <cassert>

#include "MBody1.cc"
#include "hr_time.h"
#include "utils.h" // for CHECK_CUDA_ERRORS
#include "stringUtils.h"

#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif


#define MYRAND(Y,X) Y = Y * 1103515245 +12345; X= (Y >> 16);

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

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

#define TOTAL_TME 1000000.0

CStopWatch timer;

#include "map_classol.cc"
