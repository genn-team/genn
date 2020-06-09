#include "optimiser.h"

// GeNN includes
#include "modelSpecInternal.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void calcGroupSizes(const ModelSpecInternal& model, std::vector<size_t>(&groupSizes)[CodeGenerator::OpenCL::KernelMax])
{
    using namespace CodeGenerator;
    using namespace OpenCL;

    // Loop through neuron groups
    for (const auto& n : model.getLocalNeuronGroups()) {
        // Add number of neurons to vector of neuron kernels
        groupSizes[KernelNeuronUpdate].push_back(n.second.getNumNeurons());

        // Add number of neurons to initialisation kernel (all neuron groups at least require spike counts initialising)
        groupSizes[KernelInitialize].push_back(n.second.getNumNeurons());
    }

    // Loop through synapse groups
    size_t numPreSynapseResetGroups = 0;
    for (const auto& s : model.getLocalSynapseGroups()) {
        groupSizes[KernelPresynapticUpdate].push_back(Backend::getNumPresynapticUpdateThreads(s.second));

        if (!s.second.getWUModel()->getLearnPostCode().empty()) {
            groupSizes[KernelPostsynapticUpdate].push_back(Backend::getNumPostsynapticUpdateThreads(s.second));
        }

        if (!s.second.getWUModel()->getLearnPostCode().empty()) {
            groupSizes[KernelSynapseDynamicsUpdate].push_back(Backend::getNumSynapseDynamicsThreads(s.second));
        }

        // If synapse group has individual weights and needs device initialisation
        if ((s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) && s.second.isWUVarInitRequired()) {
            const size_t numSrcNeurons = s.second.getSrcNeuronGroup()->getNumNeurons();
            const size_t numTrgNeurons = s.second.getTrgNeuronGroup()->getNumNeurons();
            if (s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                groupSizes[KernelInitializeSparse].push_back(numSrcNeurons);
            }
            else {
                groupSizes[KernelInitialize].push_back(numSrcNeurons * numTrgNeurons);
            }
        }

        // If this synapse group requires dendritic delay, it requires a pre-synapse reset
        if (s.second.isDendriticDelayRequired()) {
            numPreSynapseResetGroups++;
        }
    }

    // Add group sizes for reset kernels
    groupSizes[KernelPreNeuronReset].push_back(model.getLocalNeuronGroups().size());
    groupSizes[KernelPreSynapseReset].push_back(numPreSynapseResetGroups);
}
//--------------------------------------------------------------------------
void getOpenCLDevices(std::vector<cl::Device>& devices) {
    // Getting all platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.size() == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    // Getting all devices
    for (const auto& platform : platforms) {
        std::vector<cl::Device> platformDevices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);
        devices.insert(devices.end(), platformDevices.begin(), platformDevices.end());
    }
}
//--------------------------------------------------------------------------
int getDeviceWithMostGlobalMemory()
{
    // Get number of devices
    std::vector<cl::Device> devices;
    getOpenCLDevices(devices);
    if (devices.size() == 0) {
        throw std::runtime_error("No OpenCL devices found");
    }

    // Loop through devices
    size_t mostGlobalMemory = 0;
    int bestDevice = -1;
    for (int d = 0; d < devices.size(); d++) {
        cl_ulong deviceGlobalMemSize = devices[d].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

        // If this device improves on previous best
        if (deviceGlobalMemSize > mostGlobalMemory) {
            mostGlobalMemory = deviceGlobalMemSize;
            bestDevice = d;
        }
    }

    LOGI << "Using device " << bestDevice << " which has " << mostGlobalMemory << " bytes of global memory";
    return bestDevice;
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
Backend createBackend(const ModelSpecInternal& model, const filesystem::path& outputPath, int localHostID,
	const Preferences& preferences)
{
    // If we should select device with most memory, do so otherwise use manually selected device
    const int deviceID = (preferences.deviceSelectMethod == DeviceSelect::MOST_MEMORY)
        ? getDeviceWithMostGlobalMemory() : preferences.manualDeviceID;
    return Backend(preferences.manualWorkGroupSizes, preferences, localHostID, model.getPrecision(), deviceID);
}
}   // namespace Optimiser
}   // namespace OpenCL
}   // namespace CodeGenerator
