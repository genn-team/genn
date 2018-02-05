#pragma once


//-----------------------------------------------------------------------
/*!  \file generateInit.h

  \brief Contains functions to generate code for initialising
  kernel state variables. Part of the code generation section.
*/
//--------------------------------------------------------------------------
// Standard C++ includes
#include <string>

// Forward declarations
class NNmodel;

void genInit(const NNmodel &model,          //!< Model description
             const std::string &path);      //!< Path for code generationn
