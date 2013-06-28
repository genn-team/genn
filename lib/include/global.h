/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#ifndef _GLOBAL_H_
#define _GLOBAL_H_ //!< macro for avoiding multiple inclusion during compilation

//--------------------------------------------------------------------------
/*! \file global.h

\brief Global header file containing a few global variables. Part of the code generation section.

This global header file also takes care of including some generally used cuda support header files.
*/
//--------------------------------------------------------------------------


#include <iostream>
using namespace std; // replaced these two lines : problem with visual studio
#include <fstream>
#include <sstream>
#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cuda_runtime.h> //EY
//#include <cutil_inline.h> // does not exist anymore in CUDA 5.0
#include <helper_cuda.h>
//#include <helper_cuda_drvapi.h>

cudaDeviceProp *deviceProp; //!< Global variable that contains the detected properties of all CUDA-enabled devices
int devN; //!< Global variable that contains the number of detected CUDA-enabled devices
int optimiseBlockSize= 1; //!< Flag that regulates whether BlockSize optimisation shall be attempted

#endif  // _GLOBAL_H_
