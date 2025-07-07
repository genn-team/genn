#include "optimiser.h"

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::ISPC::Optimiser
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::ISPC::Optimiser
{
Backend createBackend(const ModelSpecInternal&,const filesystem::path&,
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

    return Backend(preferences);
}
}   // namespace GeNN::CodeGenerator::SingleThreadedCPU::Optimiser
