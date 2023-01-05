#pragma once

// GeNN includes
#include "backendExport.h"

// OpenCL backend includes
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
// GeNN::CodeGenerator::OpenCL::Optimiser
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::OpenCL::Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath, 
                                     plog::Severity backendLevel, plog::IAppender *backendAppender, 
                                     const Preferences &preferences);
}   // namespace GeNN::CodeGenerator::OpenCL::Optimiser
