/*--------------------------------------------------------------------------
  Author: Thomas Nowotny
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  T.Nowotny@sussex.ac.uk
  
  initial version: 2010-02-07
  
  --------------------------------------------------------------------------*/

//-----------------------------------------------------------------------
/*!  \file generateRunner.h

  \brief Contains functions to generate code for running the
  simulation on the GPU, and for I/O convenience functions between GPU
  and CPU space. Part of the code generation section.
*/
//--------------------------------------------------------------------------

#include "modelSpec.h"

#include <string>
#include <fstream>

using namespace std;


//--------------------------------------------------------------------------
//! \brief This fucntion generates host and device variable definitions, of the given type and name.
//--------------------------------------------------------------------------

void variable_def(ofstream &os, const string &type, const string &name);


//--------------------------------------------------------------------------
//! \brief This fucntion generates host extern variable definitions, of the given type and name.
//--------------------------------------------------------------------------

void extern_variable_def(ofstream &os, const string &type, const string &name);


//--------------------------------------------------------------------------
/*!
  \brief A function that generates predominantly host-side code.

  In this function host-side functions and other code are generated,
  including: Global host variables, "allocatedMem()" function for
  allocating memories, "freeMem" function for freeing the allocated
  memories, "initialize" for initializing host variables, "gFunc" and
  "initGRaw()" for use with plastic synapses if such synapses exist in
  the model.  
*/
//--------------------------------------------------------------------------

void genRunner(const NNmodel &model, //!< Model description
               const string &path //!< Path for code generationn
               );


//----------------------------------------------------------------------------
/*!
  \brief A function to generate the code that simulates the model on the GPU

  The function generates functions that will spawn kernel grids onto the GPU (but not the actual kernel code which is generated in "genNeuronKernel()" and "genSynpaseKernel()"). Generated functions include "copyGToDevice()", "copyGFromDevice()", "copyStateToDevice()", "copyStateFromDevice()", "copySpikesFromDevice()", "copySpikeNFromDevice()" and "stepTimeGPU()". The last mentioned function is the function that will initialize the execution on the GPU in the generated simulation engine. All other generated functions are "convenience functions" to handle data transfer from and to the GPU.
*/
//----------------------------------------------------------------------------

#ifndef CPU_ONLY
void genRunnerGPU(const NNmodel &model, //!< Model description
                  const string &path //!< Path for code generation
                  );
#endif // CPU_ONLY


//----------------------------------------------------------------------------
/*!
  \brief A function that generates the Makefile for all generated GeNN code.
*/
//----------------------------------------------------------------------------

void genMakefile(const NNmodel &model, //!< Model description
                 const string &path    //!< Path for code generation
                 );
