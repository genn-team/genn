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
#pragma once

// GeNN includes
#include "variableMode.h"

namespace GENN_PREFERENCES {    
    extern bool debugCode; //!< Request debug data to be embedded in the generated code
    extern bool mergePostsynapticModels; //!< Should compatible postsynaptic models and dendritic delay buffers be merged? This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved
    extern VarMode defaultVarMode;  //!< What is the default behaviour for model state variables? Historically, everything was allocated on both host AND device and initialised on HOST.
    extern VarMode defaultSparseConnectivityMode; //! What is the default behaviour for sparse synaptic connectivity? Historically, everything was allocated on both the host AND device and initialised on HOST
    extern bool autoRefractory; //!< Flag for signalling whether spikes are only reported if thresholdCondition changes from false to true (autoRefractory == 1) or s
}
