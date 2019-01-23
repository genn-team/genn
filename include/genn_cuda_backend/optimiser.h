#pragma once

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
Backend createBackend(const NNmodel &model, const filesystem::path &outputPath, int localHostID, 
                      const Backend::Preferences &preferences);
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator