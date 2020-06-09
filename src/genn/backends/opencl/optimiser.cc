#include "optimiser.h"

// GeNN includes
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
    
    LOGI << "Using device " << bestDevice->getInfo<CL_DEVICE_NAME>() << " which has " << bestDevice->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << " bytes of global memory";
    return (unsigned int)std::distance(devices.cbegin(), bestDevice);
}
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
Backend createBackend(const ModelSpecInternal &model, const filesystem::path &outputPath,
                      plog::Severity backendLevel, plog::IAppender *backendAppender,
                      const Preferences &preferences)
{
    // If we should select device with most memory, do so otherwise use manually selected device
    const unsigned int platformID = preferences.manualPlatformID;
    const unsigned int deviceID = (preferences.deviceSelectMethod == DeviceSelect::MOST_MEMORY)
        ? getDeviceWithMostGlobalMemory(platformID) : preferences.manualDeviceID;
    
    return Backend(preferences.manualWorkGroupSizes, preferences, model.getPrecision(), platformID, deviceID);
}
}   // namespace Optimiser
}   // namespace OpenCL
}   // namespace CodeGenerator
