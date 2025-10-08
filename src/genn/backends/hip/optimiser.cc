#include "optimiser.h"

// Standard C++ includes
#include <algorithm>
#include <numeric>

// HIP includes
#include <hip/hip_runtime.h>

// PLOG includes
#include <plog/Log.h>
// GeNN includes
#include "logging.h"
#include "modelSpecInternal.h"

// HIP backend includes
#include "utils.h"

using namespace GeNN;
using namespace GeNN::CodeGenerator;
using namespace HIP;

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::HIP::Optimiser
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::HIP::Optimiser
{
Backend createBackend(const ModelSpecInternal &model, const filesystem::path&, 
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

    // If any manual block sizes haven't been set
    KernelBlockSize manualBlockSizes = preferences.manualBlockSizes;
    if(std::any_of(manualBlockSizes.cbegin(), manualBlockSizes.cend(),
                   [](size_t s){ return (s == 0); }))
    {
        // Get manually selected device properties
        hipDeviceProp_t deviceProps;
        CHECK_HIP_ERRORS(hipGetDeviceProperties(&deviceProps, preferences.manualDeviceID));

        // Replace any zeros in block size with warp size
        std::transform(manualBlockSizes.cbegin(), manualBlockSizes.cend(), manualBlockSizes.begin(),
                       [&deviceProps](size_t s){ return (s == 0) ? deviceProps.warpSize : s; });

    } 

    return Backend(manualBlockSizes, preferences, preferences.manualDeviceID, model.zeroCopyInUse());
}
}   // namespace GeNN::CodeGenerator::HIP::Optimiser
