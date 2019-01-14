#pragma once

// Standard C++ includes
#include <string>
#include <vector>

// GeNN includes
#include "models.h"

//--------------------------------------------------------------------------
// Utils
//--------------------------------------------------------------------------
namespace Utils
{
//--------------------------------------------------------------------------
//! \brief Does the code string contain any functions requiring random number generator
//--------------------------------------------------------------------------
bool isRNGRequired(const std::string &code);

//--------------------------------------------------------------------------
//! \brief Does the model with the vectors of variable initialisers and modes require an RNG for the specified init location i.e. host or device
//--------------------------------------------------------------------------
bool isInitRNGRequired(const std::vector<Models::VarInit> &varInitialisers);
}   // namespace Utils