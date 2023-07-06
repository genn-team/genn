#pragma once

// Code generator includes
#include "code_generator/environment.h"

//---------------------------------------------------------------------------
// GeNN::CodeGenerator::StandardLibrary::FunctionTypes
//---------------------------------------------------------------------------
namespace GeNN::CodeGenerator::StandardLibrary
{
//! Get standard maths functions
const EnvironmentLibrary::Library &getMathsFunctions();

//! Get std::random based host RNG functions
const EnvironmentLibrary::Library &getHostRNGFunctions(const Type::ResolvedType &precision);
}   // namespace GeNN::CodeGenerator::StandardLibrary
