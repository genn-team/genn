#pragma once

// GeNN includes
#include "backendExport.h"

// OpenCL backend includes
#include "backend.h"

// Forward declarations
class ModelSpecInternal;
namespace plog
{
class IAppender;
}


//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
{
namespace Optimiser
{
BACKEND_EXPORT Backend createBackend(const ModelSpecInternal& model, const filesystem::path& outputPath, 
									 plog::Severity backendLevel, plog::IAppender* backendAppender,
									 const Preferences& preferences);
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator
