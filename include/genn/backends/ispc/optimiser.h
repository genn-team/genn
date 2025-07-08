#pragma once

// PLOG includes
#include <plog/Severity.h>

// GeNN includes
#include "backendExport.h"

// ISPC backend includes
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
// GeNN::CodeGenerator::ISPC::Optimiser
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::ISPC::Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath, 
                                     plog::Severity backendLevel, plog::IAppender *backendAppender,
                                     const Preferences &preferences);
}   // namespace GeNN::CodeGenerator::ISPC::Optimiser
