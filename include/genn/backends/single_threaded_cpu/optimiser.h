#pragma once

// PLOG includes
#include <plog/Severity.h>

// GeNN includes
#include "backendExport.h"

// Single-threaded CPU backend includes
#include "backend.h"

// Forward declarations
class ModelSpecInternal;
namespace plog
{
class IAppender;
}

//--------------------------------------------------------------------------
// CodeGenerator::SingleThreadedCPU::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace SingleThreadedCPU
{
namespace Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal &model, const filesystem::path &sharePath,
                                     const filesystem::path &outputPath, plog::Severity backendLevel,
                                     plog::IAppender *backendAppender, const Preferences &preferences);
}
}   // namespace SingleThreadedCPU
}   // namespace CodeGenerator
