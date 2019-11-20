#pragma once

// GeNN includes
#include "backendExport.h"

// CUDA backend includes
#include "backend.h"

//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
namespace Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath, const Preferences &preferences);
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator
