#pragma once

// GeNN includes
#include "backendExport.h"

// Single-threaded CPU backend includes
#include "backend.h"

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace SingleThreadedCPU
{
namespace Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath,
                                     int localHostID, const Preferences &preferences);
}
}   // namespace SingleThreadedCPU
}   // namespace CodeGenerator
