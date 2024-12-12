#pragma once

// Standard C++ includes
#include <iostream>

// Standard C includes
#include <cstdint>

//----------------------------------------------------------------------------
// GeNN::CodeGenerator::FeNN
//----------------------------------------------------------------------------
namespace GeNN::CodeGenerator::FeNN 
{
void disassemble(std::ostream &os, uint32_t inst);
}