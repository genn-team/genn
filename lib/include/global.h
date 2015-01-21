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
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "toString.h"
#include <stdint.h>

// switch this on to debug problems in the Blocksize logic
//#define BLOCKSZ_DEBUG

using namespace std; // replaced these two lines : problem with visual studio


// THESE WILL BE REPLACED BY VARIABLES BELOW SOON IF optimiseBlockSize == 1. THEIR INITIAL VALUES ARE SET IN generateAll.cc
int neuronBlkSz;
int synapseBlkSz;
int learnBlkSz;
cudaDeviceProp *deviceProp;
int theDev;

int hostCount; //!< Global variable containing the number of hosts within the local compute cluster
int deviceCount; //!< Global variable containing the number of CUDA devices found on this host
int optimiseBlockSize = 1; //!< Flag for signalling whether or not block size optimisation should be performed
//vector<cudaDeviceProp> deviceProp; //!< Global vector containing the properties of all CUDA-enabled devices
//vector<int> synapseBlkSz; //!< Global vector containing the optimum synapse kernel block size for each device
//vector<int> learnBlkSz; //!< Global vector containing the optimum learn kernel block size for each device
//vector<int> neuronBlkSz; //!< Global vector containing the optimum neuron kernel block size for each device
int UIntSz = sizeof(unsigned int) * 8; //!< size of the unsigned int variable type on the local architecture
int logUIntSz = (int) (logf((float) UIntSz) / logf(2.0f) + 1e-5f); //!< logarithm of the size of the unsigned int variable type on the local architecture

#endif  // _GLOBAL_H_
