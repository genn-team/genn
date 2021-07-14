#pragma once

// GeNN includes
#include "backendExport.h"

// CUDA backend includes
#include "backend.h"

// Forward declarations
class ModelSpecInternal;
namespace plog
{
class IAppender;
}


//--------------------------------------------------------------------------
// CodeGenerator::CUDA::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace CUDA
{
namespace Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath, 
                                     plog::Severity backendLevel, plog::IAppender *backendAppender, 
                                     const Preferences &preferences);
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator
