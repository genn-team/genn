/*--------------------------------------------------------------------------
   Author: Thomas Nowotny

   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK 

   email to:  T.Nowotny@sussex.ac.uk

   initial version: 2010-02-07

--------------------------------------------------------------------------*/

//------------------------------------------------------------------------
/*! \file generateKernels.h

  \brief Contains functions that generate code for CUDA kernels. Part of the code generation section.

*/
//-------------------------------------------------------------------------

#include "modelSpec.h"

#include <string>
#include <fstream>

using namespace std;


// The CPU_ONLY version does not need any of this
#ifndef CPU_ONLY


//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA kernel that simulates all neurons in the model.

  The code generated upon execution of this function is for defining GPU side global variables that will hold model state in the GPU global memory and for the actual kernel function for simulating the neurons for one time step.
*/
//-------------------------------------------------------------------------

void genNeuronKernel(const NNmodel &model, //!< Model description
                     const string &path  //!< Path for code generation
                     );


//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic 
  spikes or spike type events

*/
//-------------------------------------------------------------------------

void generate_process_presynaptic_events_code(
    ostream &os, //!< output stream for code
    const NNmodel &model, //!< the neuronal network model to generate code for
    unsigned int src, //!< the number of the src neuron population
    unsigned int trg, //!< the number of the target neuron population
    int i, //!< the index of the synapse group being processed
    const string &localID, //!< the variable name of the local ID of the thread within the synapse group
    unsigned int inSynNo, //!< the ID number of the current synapse population as the incoming population to the target neuron population
    const string &postfix //!< whether to generate code for true spikes or spike type events
                                              );

//-------------------------------------------------------------------------
/*!
  \brief Function for generating a CUDA kernel for simulating all synapses.

  This functions generates code for global variables on the GPU side that are 
  synapse-related and the actual CUDA kernel for simulating one time step of 
  the synapses.
*/
//-------------------------------------------------------------------------

void genSynapseKernel(const NNmodel &model, //!< Model description
                      const string &path //!< Path for code generation
                      );

#endif // CPU_ONLY
