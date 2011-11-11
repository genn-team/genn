/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

#ifndef _GLOBAL_H_
#define _GLOBAL_H_


using namespace std;

#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>

cudaDeviceProp *deviceProp;
int devN;
int optimiseBlockSize= 1;

#endif  // _GLOBAL_H_
