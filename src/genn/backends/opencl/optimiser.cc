#include "optimiser.h"

// GeNN includes
#include "modelSpecInternal.h"

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::Optimiser
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
{
namespace Optimiser
{
Backend createBackend(const ModelSpecInternal &model, const filesystem::path &,
                      plog::Severity backendLevel, plog::IAppender *backendAppender,
                      const Preferences &preferences)
{
    // If there isn't already a plog instance, initialise one
    if(plog::get<Logging::CHANNEL_BACKEND>() == nullptr) {
        plog::init<Logging::CHANNEL_BACKEND>(backendLevel, backendAppender);
    }
    // Otherwise, set it's max severity from GeNN preferences
    else {
        plog::get<Logging::CHANNEL_BACKEND>()->setMaxSeverity(backendLevel);
    }

	KernelWorkGroupSize workGroupSize = std::array<size_t, KernelMax>();

    return Backend(workGroupSize, preferences, "test", 1);
}
}   // namespace Optimiser
}   // namespace CUDA
}   // namespace CodeGenerator
