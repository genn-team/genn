#include "optimiser.h"

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "logging.h"
#include "modelSpecInternal.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
unsigned int getDeviceWithMostGlobalMemory(unsigned int platformID)
{
    // Get platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(platformID < platforms.size());

    // Get devices
    std::vector<cl::Device> devices;
    platforms[platformID].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // Loop through devices
    const auto bestDevice = std::max_element(devices.cbegin(), devices.cend(),
                                             [](const cl::Device &a, const cl::Device &b)
                                             {
                                                 const cl_ulong aMem = a.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                                                 const cl_ulong bMem = b.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                                                 return (aMem < bMem);
                                             });

    LOGI_BACKEND << "Using device " << bestDevice->getInfo<CL_DEVICE_NAME>() << " which has " << bestDevice->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << " bytes of global memory";
    return (unsigned int)std::distance(devices.cbegin(), bestDevice);
}
}
//--------------------------------------------------------------------------
// GeNN::CodeGenerator::OpenCL::Optimiser
//--------------------------------------------------------------------------
namespace GeNN::CodeGenerator::OpenCL::Optimiser
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

    // If we should select device with most memory, do so otherwise use manually selected device
    const unsigned int platformID = preferences.manualPlatformID;
    const unsigned int deviceID = (preferences.deviceSelectMethod == DeviceSelect::MOST_MEMORY)
        ? getDeviceWithMostGlobalMemory(platformID) : preferences.manualDeviceID;

    return Backend(preferences.manualWorkGroupSizes, preferences, model.getPrecision(), platformID, deviceID);
}
}   // namespace GeNN::CodeGenerator::OpenCL::Optimiser
