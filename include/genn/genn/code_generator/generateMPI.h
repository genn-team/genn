#pragma once

/*--------------------------------------------------------------------------
  Author: Mengchi Zhang
  
  Institute: Center for Computational Neuroscience and Robotics
  University of Sussex
  Falmer, Brighton BN1 9QJ, UK 
  
  email to:  zhan2308@purdue.edu
  
  initial version: 2017-07-19
  
  --------------------------------------------------------------------------*/

//-----------------------------------------------------------------------
/*!  \file generateMPI.h

  \brief Contains functions to generate code for running the
  simulation with MPI. Part of the code generation section.
*/
//--------------------------------------------------------------------------

// Standard C++ includes
#include <string>

// GeNN includes
#include "gennExport.h"

// Forward declarations
class ModelSpecInternal;

namespace CodeGenerator
{
class BackendBase;
class CodeStream;
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
namespace CodeGenerator
{
//! A function that generates predominantly MPI infrastructure code.
/*! In this function MPI infrastructure code are generated, including: MPI send and receive functions. */
void GENN_EXPORT generateMPI(CodeStream &os, const ModelSpecInternal &model, const BackendBase &backend, bool standaloneModules);
}
