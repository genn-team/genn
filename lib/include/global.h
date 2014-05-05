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
#include "modelSpec.h"

using namespace std;


string path; //!< Global string holding the directory in which GeNN code is to be deposited
NNmodel *model; //!< Global model object which specifies the structure and properties of the network model
int deviceCount; //!< Global variable containing the number of CUDA devices present on this host
cudaDeviceProp *deviceProp; //!< Global array listing the properties of each CUDA device on this host
int *synapseBlkSz; //!< Global array containing the optimum synapse kernel block size for each device
int *learnBlkSz; //!< Global array containing the optimum learn kernel block size for each device
int *neuronBlkSz; //!< Global array containing the optimum neuron kernel block size for each device
int UIntSz = sizeof(unsigned int) * 8; //!< size of the unsigned int variable type on the local architecture
int logUIntSz = (int) (logf((float) UIntSz) / logf(2.0f) + 1e-5f); //!< logarithm of the size of the unsigned int variable type on the local architecture
int useAllHosts = 0; //!< Global flag indicating that all cluster nodes should be utilised
int useAllCudaDevices = 0; //!< Global flag indicating that all CUDA devices should be utilised
int optCudaBlockSize = 1; //!< Global flag indicating that block size optimisation should be performed

#endif  // _GLOBAL_H_
