/*--------------------------------------------------------------------------
  Author: Thomas Nowotny
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  T.Nowotny@sussex.ac.uk
  
  initial version: 2010-02-07
  
  --------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generateCPU.h

  \brief Functions for generating code that will run the neuron and synapse simulations on the CPU. Part of the code generation section.

*/
//--------------------------------------------------------------------------

#include "modelSpec.h"

#include <string>
#include <fstream>

using namespace std;


//--------------------------------------------------------------------------
/*!
  \brief Function that generates the code of the function the will simulate all neurons on the CPU.
*/
//--------------------------------------------------------------------------

void genNeuronFunction(const NNmodel &model, //!< Model description
                       const string &path //!< Path for code generation
                       );


//-------------------------------------------------------------------------
/*!
  \brief Function for generating the CUDA synapse kernel code that handles presynaptic 
  spikes or spike type events
*/
//-------------------------------------------------------------------------

void generate_process_presynaptic_events_code_CPU(
    ostream &os, //!< output stream for code
    const string &sgName,
    const SynapseGroup &sg,
    const string &postfix, //!< whether to generate code for true spikes or spike type events
    const string &ftype);


//--------------------------------------------------------------------------
/*!
  \brief Function that generates code that will simulate all synapses of the model on the CPU.
*/
//--------------------------------------------------------------------------

void genSynapseFunction(const NNmodel &model, //!< Model description
                        const string &path //!< Path for code generation
                        );
