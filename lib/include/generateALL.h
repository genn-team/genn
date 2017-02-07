/*--------------------------------------------------------------------------
   Author: Thomas Nowotny

   Institute: Center for Computational Neuroscience and Robotics
              University of Sussex
              Falmer, Brighton BN1 9QJ, UK

   email to:  T.Nowotny@sussex.ac.uk

   initial version: 2010-02-07

--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file generateALL.cc

  \brief Main file combining the code for code generation. Part of the code generation section.

  The file includes separate files for generating kernels (generateKernels.cc), generating the CPU side code for running simulations on either the CPU or GPU (generateRunner.cc) and for CPU-only simulation code (generateCPU.cc).

*/
//--------------------------------------------------------------------------

#include "modelSpec.h"

#include <string>

using namespace std;


//--------------------------------------------------------------------------
/*! \brief This function will call the necessary sub-functions to generate the code for simulating a model.
 */
//--------------------------------------------------------------------------

void generate_model_runner(const NNmodel &model,  //!< Model description
                           const string &path      //!< Path where the generated code will be deposited
                           );


//--------------------------------------------------------------------------
/*!
  \brief Helper function that prepares data structures and detects the hardware properties to enable the code generation code that follows.

  The main tasks in this function are the detection and characterization of the GPU device present (if any), choosing which GPU device to use, finding and appropriate block size, taking note of the major and minor version of the CUDA enabled device chosen for use, and populating the list of standard neuron models. The chosen device number is returned.
*/
//--------------------------------------------------------------------------

#ifndef CPU_ONLY
void chooseDevice(NNmodel &model, //!< the nn model we are generating code for
                  const string &path     //!< path the generated code will be deposited
                  );
#endif
