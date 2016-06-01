/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
	      Falmer, Brighton BN1 9QJ, UK 
  
   email to:  T.Nowotny@sussex.ac.uk
  
   initial version: 2010-02-07
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file global.h

\brief Global header file containing a few global variables. Part of the code generation section.
*/
//--------------------------------------------------------------------------

#ifndef GLOBAL_CC
#define GLOBAL_CC

#include "global.h"


namespace GENN_FLAGS {
    unsigned int calcSynapseDynamics= 0;
    unsigned int calcSynapses= 1;
    unsigned int learnSynapsesPost= 2;
    unsigned int calcNeurons= 3;
};

namespace GENN_PREFERENCES {    
    int optimiseBlockSize = 1; //!< Flag for signalling whether or not block size optimisation should be performed
    int autoChooseDevice= 1; //!< Flag to signal whether the GPU device should be chosen automatically 
    bool optimizeCode = false; //!< Request speed-optimized code, at the expense of floating-point accuracy
    bool debugCode = false; //!< Request debug data to be embedded in the generated code
    bool showPtxInfo = false; //!< Request that PTX assembler information be displayed for each CUDA kernel during compilation
    double asGoodAsZero = 1e-19; //!< Global variable that is used when detecting close to zero values, for example when setting sparse connectivity from a dense matrix
    int defaultDevice= 0; //! default GPU device; used to determine which GPU to use if chooseDevice is 0 (off)
    unsigned int neuronBlockSize= 32;
    unsigned int synapseBlockSize= 32;
    unsigned int learningBlockSize= 32;
    unsigned int synapseDynamicsBlockSize= 32;
    unsigned int autoRefractory= 1; //!< Flag for signalling whether spikes are only reported if thresholdCondition changes from false to true (autoRefractory == 1) or spikes are emitted whenever thresholdCondition is true no matter what.
};

// These will eventually go inside e.g. some HardwareConfig class. Putting them here meanwhile.
int neuronBlkSz; //!< Global variable containing the GPU block size for the neuron kernel
int synapseBlkSz; //!< Global variable containing the GPU block size for the synapse kernel
int learnBlkSz; //!< Global variable containing the GPU block size for the learn kernel
int synDynBlkSz; //!< Global variable containing the GPU block size for the synapse dynamics kernel
//vector<cudaDeviceProp> deviceProp; //!< Global vector containing the properties of all CUDA-enabled devices
//vector<int> synapseBlkSz; //!< Global vector containing the optimum synapse kernel block size for each device
//vector<int> learnBlkSz; //!< Global vector containing the optimum learn kernel block size for each device
//vector<int> neuronBlkSz; //!< Global vector containing the optimum neuron kernel block size for each device
//vector<int> synDynBlkSz; //!< Global vector containing the optimum synapse dynamics kernel block size for each device
#ifndef CPU_ONLY
cudaDeviceProp *deviceProp;
int theDevice; //!< Global variable containing the currently selected CUDA device's number
int deviceCount; //!< Global variable containing the number of CUDA devices on this host
#endif
int hostCount; //!< Global variable containing the number of hosts within the local compute cluster

#endif // GLOBAL_CC
