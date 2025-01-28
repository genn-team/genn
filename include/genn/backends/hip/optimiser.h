#pragma once

// GeNN includes
#include "backendExport.h"

// CUDA backend includes
#include "backend.h"

// Forward declarations
namespace GeNN
{
class ModelSpecInternal;
}

namespace plog
{
class IAppender;
}


//--------------------------------------------------------------------------
// GeNN::CodeGenerator::HIP::Optimiser
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::HIP::Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath, 
                                     plog::Severity backendLevel, plog::IAppender *backendAppender, 
                                     const Preferences &preferences);
}   // namespace GeNN::CodeGenerator::HIP::Optimiser
