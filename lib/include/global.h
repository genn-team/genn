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

#ifndef CPU_ONLY
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "toString.h"
#include <stdint.h>

using namespace std; // replaced these two lines : problem with visual studio

// THESE WILL BE REPLACED BY VARIABLES BELOW SOON IF optimiseBlockSize == 1. THEIR INITIAL VALUES ARE SET IN generateAll.cc
int neuronBlkSz;
int synapseBlkSz;
int learnBlkSz;
int synDynBlkSz;
#ifndef CPU_ONLY
cudaDeviceProp *deviceProp;
int theDev;
#endif

int hostCount; //!< Global variable containing the number of hosts within the local compute cluster
int deviceCount; //!< Global variable containing the number of CUDA devices found on this host
//vector<cudaDeviceProp> deviceProp; //!< Global vector containing the properties of all CUDA-enabled devices
//vector<int> synapseBlkSz; //!< Global vector containing the optimum synapse kernel block size for each device
//vector<int> learnBlkSz; //!< Global vector containing the optimum learn kernel block size for each device
//vector<int> neuronBlkSz; //!< Global vector containing the optimum neuron kernel block size for each device

int UIntSz = sizeof(unsigned int) * 8; //!< size of the unsigned int variable type on the local architecture
int logUIntSz = (int) (logf((float) UIntSz) / logf(2.0f) + 1e-5f); //!< logarithm of the size of the unsigned int variable type on the local architecture

namespace GENN_FLAGS {
    unsigned int COPY= 1;
    unsigned int NOCOPY= ~COPY;

    unsigned int calcSynapseDynamics= 0;
    unsigned int calcSynapses= 1;
    unsigned int learnSynapsesPost= 2;
    unsigned int calcNeurons= 3;
};

namespace GENN_PREFERENCES {    
    int optimiseBlockSize = 1; //!< Flag for signalling whether or not block size optimisation should be performed
    int chooseDevice= 1; //!< Flag to signal whether the GPU device should be chosen automatically 
    double asGoodAsZero = 1e-19; //!< Global variable that is used when detecting close to zero values, for example when setting sparse connectivity from a dense matrix
    int defaultDevice= 0; //! default GPU device; used to determine which GPU to use if chooseDevice is 0 (off)

    unsigned int neuronBlockSize= 32;
    unsigned int synapseBlockSize= 32;
    unsigned int learningBlockSize= 32;
    unsigned int synapseDynamicsBlockSize= 32;

    unsigned int autoRefractory= 1; //!< Flag for signalling whether spikes are only reported if thresholdCondition changes from false to true (autoRefractory == 1) or spikes are emitted whenever thresholdCondition is true no matter what.
};
    

#endif  // _GLOBAL_H_
