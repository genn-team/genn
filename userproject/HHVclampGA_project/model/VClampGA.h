/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file classol_sim.h

\brief Header file containing global variables and macros used in running the HHVClamp/VClampGA model.
*/
//--------------------------------------------------------------------------

#include <cassert>
using namespace std;
#include "hr_time.cpp"

#include "utils.h" // for CHECK_CUDA_ERRORS
#include <cuda_runtime.h>
#include "HHVClamp.cc"
#include "HHVClamp_CODE/runner.cc"
#include "../../../lib/include/numlib/randomGen.h"
#include "../../../lib/include/numlib/gauss.h"
randomGen R;
randomGauss RG;
#include "helper.h"

//#define DEBUG_PROCREATE
#include "GA.cc"

#define RAND(Y,X) Y = Y * 1103515245 +12345;X= (unsigned int)(Y >> 16) & 32767

// and some global variables
double t= 0.0f;
unsigned int iT= 0;
CStopWatch timer;
