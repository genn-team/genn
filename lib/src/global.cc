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
#include "global.h"

namespace GENN_PREFERENCES {    
    bool mergePostsynapticModels = false; //!< Should compatible postsynaptic models and dendritic delay buffers be merged? This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved
    VarLocation defaultVarLocation = VarLocation::LOC_HOST_DEVICE;  //!< What is the default behaviour for model state variables? Historically, everything was allocated on both host AND device and initialised on HOST.
    VarLocation defaultSparseConnectivityLocation = VarLocation::LOC_HOST_DEVICE;   //! What is the default behaviour for sparse synaptic connectivity? Historically, everything was allocated on both the host AND device and initialised on HOST
    bool autoRefractory= true; //!< Flag for signalling whether spikes are only reported if thresholdCondition changes from false to true (autoRefractory == 1) or spikes are emitted whenever thresholdCondition is true no matter what.
};