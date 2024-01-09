#pragma once

// GeNN includes
#include "gennExport.h"

// Code generator includes
#include "code_generator/environment.h"

//---------------------------------------------------------------------------
// GeNN::CodeGenerator::StandardLibrary::FunctionTypes
//---------------------------------------------------------------------------
namespace GeNN::CodeGenerator::StandardLibrary
{
//! Get standard maths functions
GENN_EXPORT const EnvironmentLibrary::Library &getMathsFunctions();

//! Get std::random based host RNG functions
GENN_EXPORT const EnvironmentLibrary::Library &getHostRNGFunctions(const Type::ResolvedType &precision);
}   // namespace GeNN::CodeGenerator::StandardLibrary
