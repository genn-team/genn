#include "backend.h"

// Standard C++ includes
#include <algorithm>

// PLOG includes
#include <plog/Log.h>

// GeNN includes
#include "gennUtils.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/substitutions.h"
#include "code_generator/codeGenUtils.h"

// OpenCL backend includes
#include "utils.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace {

//--------------------------------------------------------------------------
// Timer
//--------------------------------------------------------------------------
class Timer {
public:
	// TO BE REVIEWED
	Timer(CodeGenerator::CodeStream& codeStream, const std::string& name, bool timingEnabled, bool synchroniseOnStop = false)
	:	m_CodeStream(codeStream), m_Name(name), m_TimingEnabled(timingEnabled), m_SynchroniseOnStop(synchroniseOnStop) {
		
	}
private:
	//--------------------------------------------------------------------------
	// Members
	//--------------------------------------------------------------------------
	CodeGenerator::CodeStream& m_CodeStream;
	const std::string m_Name;
	const bool m_TimingEnabled;
	const bool m_SynchroniseOnStop;
};
}

namespace CodeGenerator
{
namespace OpenCL
{
const char* Backend::KernelNames[KernelMax] = {
	"updateNeuronsKernel",
	"updatePresynapticKernel",
	"updatePostsynapticKernel",
	"updateSynapseDynamicsKernel",
	"initializeKernel",
	"initializeSparseKernel",
	"preNeuronResetKernel",
	"preSynapseResetKernel" };
//--------------------------------------------------------------------------
Backend::Backend(const KernelWorkGroupSize& kernelWorkGroupSizes, const Preferences& preferences,
	int localHostID, const std::string& scalarType, int device)
	: BackendBase(localHostID, scalarType), m_KernelWorkGroupSizes(kernelWorkGroupSizes), m_Preferences(preferences), m_ChosenDeviceID(device)
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::Backend");
}
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream& os, const ModelSpecInternal& model, NeuronGroupSimHandler simHandler, NeuronGroupHandler wuVarUpdateHandler) const
{
	// Generate reset kernel to be run before the neuron kernel

	// Collect the arguments to be sent to the kernel
	std::vector<std::string> kernelArgs;
	// Loop through remote neuron groups
	for (const auto& n : model.getRemoteNeuronGroups()) {
		if (n.second.hasOutputToHost(getLocalHostID()) && n.second.isDelayRequired()) {
			kernelArgs.push_back("d_spkQuePtr" + n.first);
		}
	}
	// Loop through local neuron groups
	for (const auto& n : model.getLocalNeuronGroups()) {
		if (n.second.isDelayRequired()) { // with delay
			if (std::find(kernelArgs.begin(), kernelArgs.end(), "d_spkQuePtr" + n.first) == kernelArgs.end()) {
				kernelArgs.push_back("d_spkQuePtr");
			}
		}
		if (n.second.isSpikeEventRequired()) {
			kernelArgs.push_back("d_glbSpkCntEvnt" + n.first);
		}
		kernelArgs.push_back("d_glbSpkCnt" + n.first);
	}
	std::string allKernelArgs = "";
	for (int i = 0; i < kernelArgs.size(); i++) {
		allKernelArgs += "__global unsigned int* ";
		if (i == (kernelArgs.size() - 1)) {
			allKernelArgs += kernelArgs[i];
		} else {
			allKernelArgs += kernelArgs[i] + ", ";
		}
	}
	// ********************

	// Actual kernel generation
	size_t idPreNeuronReset = 0;
	os << "extern \"C\" const char* " << KernelNames[KernelPreNeuronReset] << "Src = R\"(typedef float scalar;" << std::endl;
	os << "__kernel void " << KernelNames[KernelPreNeuronReset] << "(" << allKernelArgs << ")";
	{
		CodeStream::Scope b(os);

		os << "size_t groupId = get_group_id(0);" << std::endl;
		os << "size_t localId = get_local_id(0);" << std::endl;
		os << "unsigned int id = " << m_KernelWorkGroupSizes[KernelPreNeuronReset] << " * groupId + localId;" << std::endl;

		// Loop through remote neuron groups
		for (const auto& n : model.getRemoteNeuronGroups()) {
			if (n.second.hasOutputToHost(getLocalHostID()) && n.second.isDelayRequired()) {
				if (idPreNeuronReset > 0) {
					os << "else ";
				}
				os << "if(id == " << (idPreNeuronReset++) << ")";
				{
					CodeStream::Scope b(os);
					os << "d_spkQuePtr" << n.first << " = (d_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
				}
			}
		}

		// Loop through local neuron groups
		for (const auto& n : model.getLocalNeuronGroups()) {
			if (idPreNeuronReset > 0) {
				os << "else ";
			}
			os << "if(id == " << (idPreNeuronReset++) << ")";
			{
				CodeStream::Scope b(os);

				if (n.second.isDelayRequired()) { // with delay
					os << "d_spkQuePtr" << n.first << " = (d_spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;

					if (n.second.isSpikeEventRequired()) {
						os << "d_glbSpkCntEvnt" << n.first << "[d_spkQuePtr" << n.first << "] = 0;" << std::endl;
					}
					if (n.second.isTrueSpikeRequired()) {
						os << "d_glbSpkCnt" << n.first << "[d_spkQuePtr" << n.first << "] = 0;" << std::endl;
					}
					else {
						os << "d_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
					}
				}
				else { // no delay
					if (n.second.isSpikeEventRequired()) {
						os << "d_glbSpkCntEvnt" << n.first << "[0] = 0;" << std::endl;
					}
					os << "d_glbSpkCnt" << n.first << "[0] = 0;" << std::endl;
				}
			}
		}
	}
	// Closing the multiline char*
	os << ")\";" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream& os, const ModelSpecInternal& model,
	SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler, SynapseGroupHandler wumEventHandler,
	SynapseGroupHandler postLearnHandler, SynapseGroupHandler synapseDynamicsHandler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genSynapseUpdate");
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream &os, const ModelSpecInternal &model,
                      NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                      SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                      SynapseGroupHandler sgSparseInitHandler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genInit");
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream& os) const
{
	os << "// Standard C++ includes" << std::endl;
	os << "#include <string>" << std::endl;
	os << "#include <stdexcept>" << std::endl;
	os << std::endl;
	os << "// Standard C includes" << std::endl;
	os << "#include <cstdint>" << std::endl;
	os << std::endl;
	os << "// OpenCL includes" << std::endl;
	os << "#define CL_USE_DEPRECATED_OPENCL_1_2_APIS" << std::endl;
	os << "#include <CL/cl.hpp>" << std::endl;
	os << std::endl;
	os << "#define DEVICE_INDEX " << m_ChosenDeviceID << std::endl;
	os << std::endl;
	os << "// ------------------------------------------------------------------------" << std::endl;
	os << "// Helper macro for error-checking OpenCL calls" << std::endl;
	os << "#define CHECK_OPENCL_ERRORS(call) {\\" << std::endl;
	os << "    cl_int error = call;\\" << std::endl;
	os << "    if (error != CL_SUCCESS) {\\" << std::endl;
	os << "        throw std::runtime_error(__FILE__\": \" + std::to_string(__LINE__) + \": cuda error \" + std::to_string(error) + \": \" + clGetErrorString(error));\\" << std::endl;
	os << "    }\\" << std::endl;
	os << "}" << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsInternalPreamble(CodeStream& os) const
{
	// Declaration of OpenCL functions
	os << "// ------------------------------------------------------------------------" << std::endl;
	os << "// OpenCL functions declaration" << std::endl;
	os << "// ------------------------------------------------------------------------" << std::endl;
	os << "namespace opencl";
	{
		CodeStream::Scope b(os);
		os << "void setUpContext(cl::Context& context, cl::Device& device, const int deviceIndex);" << std::endl;
		os << "void createProgram(const char* kernelSource, cl::Program& program, cl::Context& context);" << std::endl;
	}
	os << std::endl;
	// Declaration of OpenCL variables
	os << "extern \"C\"";
	{
		CodeStream::Scope b(os);
		os << "// OpenCL variables" << std::endl;
		os << "EXPORT_VAR cl::Context clContext;" << std::endl;
		os << "EXPORT_VAR cl::Device clDevice;" << std::endl;
		os << "EXPORT_VAR cl::Program initProgram;" << std::endl;
		os << "EXPORT_VAR cl::Program unProgram;" << std::endl;
		os << "EXPORT_VAR cl::CommandQueue commandQueue;" << std::endl;
		os << std::endl;
		os << "// OpenCL kernels" << std::endl;
		os << "EXPORT_VAR cl::Kernel initKernel;" << std::endl;
		os << "EXPORT_VAR cl::Kernel preNeuronResetKernel;" << std::endl;
		os << "EXPORT_VAR cl::Kernel updateNeuronsKernel;" << std::endl;
	}
	os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream& os) const
{
	// Generating OpenCL variables for the runner
	os << "extern \"C\"";
	{
		CodeStream::Scope b(os);
		os << "// OpenCL variables" << std::endl;
		os << "cl::Context clContext;" << std::endl;
		os << "cl::Device clDevice;" << std::endl;
		os << "cl::Program initProgram;" << std::endl;
		os << "cl::Program unProgram;" << std::endl;
		os << "cl::CommandQueue commandQueue;" << std::endl;
		os << std::endl;
		os << "// OpenCL kernels" << std::endl;
		os << "cl::Kernel initKernel;" << std::endl;
		os << "cl::Kernel preNeuronResetKernel;" << std::endl;
		os << "cl::Kernel updateNeuronsKernel;" << std::endl;
	}

	// Implementation of OpenCL functions declared in definitionsInternal
	os << "// ------------------------------------------------------------------------" << std::endl;
	os << "// OpenCL functions implementation" << std::endl;
	os << "// ------------------------------------------------------------------------" << std::endl;
	os << std::endl;
	os << "// Initialize context with the given device" << std::endl;
	os << "void opencl::setUpContext(cl::Context& context, cl::Device& device, const int deviceIndex)";
	{
		CodeStream::Scope b(os);
		os << "// Getting all platforms to gather devices from" << std::endl;
		os << "std::vector<cl::Platform> platforms;" << std::endl;
		os << "cl::Platform::get(&platforms); // Gets all the platforms" << std::endl;
		os << std::endl;
		os << "assert(platforms.size() > 0);" << std::endl;
		os << std::endl;
		os << "// Getting all devices and putting them into a single vector" << std::endl;
		os << "std::vector<cl::Device> devices;" << std::endl;
		os << "for (int i = 0; i < platforms.size(); i++)";
		{
			CodeStream::Scope b(os);
			os << "std::vector<cl::Device> platformDevices;" << std::endl;
			os << "platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platformDevices);" << std::endl;
			os << "devices.insert(devices.end(), platformDevices.begin(), platformDevices.end());" << std::endl;
		}
		os << std::endl;
		os << "assert(devices.size() > 0);" << std::endl;
		os << std::endl;
		os << "// Check if the device exists at the given index" << std::endl;
		os << "if (deviceIndex >= devices.size())";
		{
			CodeStream::Scope b(os);
			os << "assert(deviceIndex >= devices.size());" << std::endl;
			os << "device = devices.front();" << std::endl;
		}
		os << "else";
		{
			CodeStream::Scope b(os);
			os << "device = devices[deviceIndex]; // We will perform our operations using this device" << std::endl;
		}
		os << std::endl;
		os << "context = cl::Context(device);";
		os << std::endl;
	}
	os << std::endl;
	os << "// Create OpenCL program with the specified device" << std::endl;
	os << "void opencl::createProgram(const char* kernelSource, cl::Program& program, cl::Context& context)";
	{
		CodeStream::Scope b(os);
		os << "// Reading the kernel source for execution" << std::endl;
		os << "program = cl::Program(context, kernelSource, true);" << std::endl;
		os << "program.build(\"-cl-std=CL1.2\");" << std::endl;
	}
	os << std::endl;
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream& os, const ModelSpecInternal& model) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genAllocateMemPreamble");
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream& os, const ModelSpecInternal& model) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genStepTimeFinalisePreamble");
}
//--------------------------------------------------------------------------
void Backend::genVariableDefinition(CodeStream& definitions, CodeStream& definitionsInternal, const std::string& type, const std::string& name, VarLocation loc) const
{
	const bool deviceType = isDeviceType(type);

	if(loc & VarLocation::HOST) {
		if (deviceType) {
			throw std::runtime_error("Variable '" + name + "' is of device-only type '" + type + "' but is located on the host");
		}
		definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
	}
	if(loc & VarLocation::DEVICE) {
		definitionsInternal << "EXPORT_VAR cl::Buffer" << " d_" << name << ";" << std::endl;
	}
}
//--------------------------------------------------------------------------
void Backend::genVariableImplementation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc) const
{
	if(loc & VarLocation::HOST) {
		os << type << " " << name << ";" << std::endl;
	}
	if(loc & VarLocation::DEVICE) {
		os << "cl::Buffer" << " d_" << name << ";" << std::endl;
	}
}
//--------------------------------------------------------------------------
MemAlloc Backend::genVariableAllocation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc, size_t count) const
{
	auto allocation = MemAlloc::host(count * getSize(type));
	os << name << " = " << "(" << type << "*)malloc(" << count << " * sizeof(" << type << "));" << std::endl;
    return allocation;
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream& os, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genVariableFree");
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamDefinition(CodeStream& definitions, const std::string& type, const std::string& name, VarLocation loc) const
{
	if (loc & VarLocation::HOST) {
		definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
	}
	if (loc & VarLocation::DEVICE && ::Utils::isTypePointer(type)) {
		definitions << "EXPORT_VAR cl::Buffer" << " d_" << name << ";" << std::endl;
	}
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamImplementation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genExtraGlobalParamImplementation");
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamAllocation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genExtraGlobalParamAllocation");
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPush(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genExtraGlobalParamPush");
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamPull(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genExtraGlobalParamPull");
}
//--------------------------------------------------------------------------
void Backend::genPopVariableInit(CodeStream& os, VarLocation, const Substitutions& kernelSubs, Handler handler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genPopVariableInit");
}
//--------------------------------------------------------------------------
void Backend::genVariableInit(CodeStream& os, VarLocation, size_t, const std::string& countVarName,
	const Substitutions& kernelSubs, Handler handler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genVariableInit");
}
//--------------------------------------------------------------------------
void Backend::genSynapseVariableRowInit(CodeStream& os, VarLocation, const SynapseGroupInternal& sg,
	const Substitutions& kernelSubs, Handler handler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genSynapseVariableRowInit");
}
//--------------------------------------------------------------------------
void Backend::genVariablePush(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc, bool autoInitialized, size_t count) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genVariablePush");
}
//--------------------------------------------------------------------------
void Backend::genVariablePull(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc, size_t count) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genVariablePull");
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePush(CodeStream& os, const NeuronGroupInternal& ng, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genCurrentVariablePush");
}
//--------------------------------------------------------------------------
void Backend::genCurrentVariablePull(CodeStream& os, const NeuronGroupInternal& ng, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genCurrentVariablePull");
}
//--------------------------------------------------------------------------
MemAlloc Backend::genGlobalRNG(CodeStream& definitions, CodeStream& definitionsInternal, CodeStream& runner, CodeStream& allocations, CodeStream& free, const ModelSpecInternal&) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genGlobalRNG");
	return MemAlloc::zero();
}
//--------------------------------------------------------------------------
MemAlloc Backend::genPopulationRNG(CodeStream& definitions, CodeStream& definitionsInternal, CodeStream& runner, CodeStream& allocations, CodeStream& free,
	const std::string& name, size_t count) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genPopulationRNG");
	return MemAlloc::zero();
}
//--------------------------------------------------------------------------
void Backend::genTimer(CodeStream&, CodeStream& definitionsInternal, CodeStream& runner, CodeStream& allocations, CodeStream& free,
	CodeStream& stepTimeFinalise, const std::string& name, bool updateInStepTime) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genTimer");
}
//--------------------------------------------------------------------------
void Backend::genMakefilePreamble(std::ostream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMakefilePreamble");
}
//--------------------------------------------------------------------------
void Backend::genMakefileLinkRule(std::ostream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMakefileLinkRule");
}
//--------------------------------------------------------------------------
void Backend::genMakefileCompileRule(std::ostream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMakefileCompileRule");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildConfigProperties(std::ostream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMSBuildConfigProperties");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportProps(std::ostream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMSBuildImportProps");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildItemDefinitions(std::ostream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMSBuildItemDefinitions");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildCompileModule(const std::string& moduleName, std::ostream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMSBuildCompileModule");
}
//--------------------------------------------------------------------------
void Backend::genMSBuildImportTarget(std::ostream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMSBuildImportTarget");
}
//--------------------------------------------------------------------------
bool Backend::isGlobalRNGRequired(const ModelSpecInternal& model) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::isGlobalRNGRequired");
	return false;
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePull(CodeStream& os, const NeuronGroupInternal& ng, bool spikeEvent) const
{
	printf("\nTO BE IMPLEMENTED: CodeGenerator::OpenCL::Backend::genCurrentSpikePull");
}
//--------------------------------------------------------------------------
void Backend::genCurrentSpikePush(CodeStream& os, const NeuronGroupInternal& ng, bool spikeEvent) const
{
	printf("\nTO BE IMPLEMENTED: CodeGenerator::OpenCL::Backend::genCurrentSpikePush");
}
//--------------------------------------------------------------------------
void Backend::addDeviceType(const std::string& type, size_t size)
{
	addType(type, size);
	m_DeviceTypes.emplace(type);
}
//--------------------------------------------------------------------------
bool Backend::isDeviceType(const std::string& type) const
{
	// Get underlying type
	const std::string underlyingType = ::Utils::isTypePointer(type) ? ::Utils::getUnderlyingType(type) : type;

	// Return true if it is in device types set
	return (m_DeviceTypes.find(underlyingType) != m_DeviceTypes.cend());
}
} // namespace OpenCL
} // namespace CodeGenerator