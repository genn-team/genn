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

using namespace std; // replaced these two lines : problem with visual studio

string path; //!< Global string holding the directory in which GeNN code is to be deposited
ostream mos = cerr; //!< output stream for debugging messages
int UIntSz = sizeof(unsigned int) * 8; //!< size of the unsigned int variable type on the local architecture
int logUIntSz = (int) (logf((float) UIntSz) / logf(2.0f) + 1e-5f); //!< logarithm of the size of the unsigned int variaint deviceCount = 0; //!< Global variable containing the number of CUDA devices present on this host
int useAllCudaDevices = 0; //!< Flag that signals whether or not all CUDA devices should be utilised
int optCudaBlockSize = 1; //!< Flag that signals whether or not block size optimisation should be performed
ble type on the local architecture
NNmodel *model; //!< Global object which specifies the structure and properties of the network model
cudaDeviceProp *deviceProp; //!< Global array containing the properties of all CUDA-enabled devices
int *synapseBlkSz; //!< Global array containing the optimum synapse kernel block size for each device
int *learnBlkSz; //!< Global array containing the optimum learn kernel block size for each device
int *neuronBlkSz; //!< Global array containing the optimum neuron kernel block size for each device

#endif  // _GLOBAL_H_
