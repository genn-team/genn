#pragma once

// PLOG includes
#include <plog/Severity.h>

// GeNN includes
#include "backendExport.h"

// Single-threaded CPU backend includes
#include "backend.h"

namespace plog
{
class IAppender;
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::SingleThreadedCPU::Optimiser
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::SingleThreadedCPU::Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath, 
                                     plog::Severity backendLevel, plog::IAppender *backendAppender,
                                     const Preferences &preferences);
}   // namespace GeNN::CodeGenerator::SingleThreadedCPU::Optimiser
