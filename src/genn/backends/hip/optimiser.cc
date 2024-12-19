#include "optimiser.h"

// Standard C++ includes
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <tuple>

// Standard C includes
#include <cstdlib>

// HIP includes
#include <hip/hip_runtime.h>

// PLOG includes
#include <plog/Log.h>

// Filesystem includes
#include "path.h"

// GeNN includes
#include "logging.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/codeGenUtils.h"
#include "code_generator/generateModules.h"
#include "code_generator/generateRunner.h"
#include "code_generator/modelSpecMerged.h"

// CUDA backend includes
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

    return Backend(preferences.manualBlockSizes, preferences, preferences.manualDeviceID, model.zeroCopyInUse());
}
}   // namespace GeNN::CodeGenerator::HIP::Optimiser
