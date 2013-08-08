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
#include <cutil.h>
#include "hr_time.h"

#define DT 0.5
#define RAND(Y,X) Y = Y * 1103515245 +12345;X= (unsigned int)(Y >> 16) & 32767

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

#define DBG_SIZE 10000

// and some global variables
float t= 0.0f;
unsigned int iT= 0;

#define PATTERNNO 100
#define INPUTBASERATE 17
//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 1000.0
#define SYN_OUT_TME 20000.0
// reset input every 100 steps == 50ms
#define PAT_TIME 50.0
#define PAT_SETTIME 200
// pattern goes off at 2 steps == 1 ms
#define PATFTIME 1.0
#define PAT_FIRETIME 2

#define TOTAL_TME 500000

//#define RAND(Y,X) Y = Y * 1103515245 +12345; X= (unsigned int)(Y / 65536) % 32768;

CStopWatch timer;

#include "hr_time.cpp"
#include "map_classol.h"
#include "map_classol.cc"
