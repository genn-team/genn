#include "backend.h"

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
	const std::string& scalarType, int device)
	: BackendBase(scalarType), m_KernelWorkGroupSizes(kernelWorkGroupSizes), m_Preferences(preferences), m_ChosenDeviceID(device)
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::Backend");
	addDeviceType("cl::Buffer", sizeof(cl::Buffer));
}
//--------------------------------------------------------------------------
void Backend::genNeuronUpdate(CodeStream& os, const ModelSpecMerged& modelMerged,
	NeuronGroupSimHandler simHandler, NeuronGroupMergedHandler wuVarUpdateHandler,
	HostHandler pushEGPHandler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genNeuronUpdate");
}
//--------------------------------------------------------------------------
void Backend::genSynapseUpdate(CodeStream& os, const ModelSpecMerged& modelMerged,
	SynapseGroupMergedHandler wumThreshHandler, SynapseGroupMergedHandler wumSimHandler,
	SynapseGroupMergedHandler wumEventHandler, SynapseGroupMergedHandler wumProceduralConnectHandler,
	SynapseGroupMergedHandler postLearnHandler, SynapseGroupMergedHandler synapseDynamicsHandler,
	HostHandler pushEGPHandler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genSynapseUpdate");
}
//--------------------------------------------------------------------------
void Backend::genInit(CodeStream& os, const ModelSpecMerged& modelMerged,
	NeuronGroupMergedHandler localNGHandler, SynapseGroupMergedHandler sgDenseInitHandler,
	SynapseGroupMergedHandler sgSparseConnectHandler, SynapseGroupMergedHandler sgSparseInitHandler,
	HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genInit");
}
//--------------------------------------------------------------------------
size_t Backend::getSynapticMatrixRowStride(const SynapseGroupInternal& sg) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::getSynapticMatrixRowStride");
	size_t x = 12;
	return x;
}
//--------------------------------------------------------------------------
void Backend::genDefinitionsPreamble(CodeStream& os, const ModelSpecMerged&) const
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
void Backend::genDefinitionsInternalPreamble(CodeStream& os, const ModelSpecMerged&) const
{
	// Generating inline OpenCL kernels
	os << "const char * initKernelSrc = \"typedef float scalar;\\" << std::endl;
	os << "\\" << std::endl;
	os << "__kernel void initializeKernel(const unsigned int deviceRNGSeed,\\" << std::endl;
	os << "__global unsigned int* glbSpkCntNeurons,\\" << std::endl;
	os << "__global unsigned int* glbSpkNeurons,\\" << std::endl;
	os << "__global scalar* VNeurons,\\" << std::endl;
	os << "__global scalar* UNeurons) {\\" << std::endl;
	os << "    int groupId = get_group_id(0);\\" << std::endl;
	os << "    int localId = get_local_id(0);\\" << std::endl;
	os << "    const unsigned int id = 32 * groupId + localId;\\" << std::endl;
	os << "\\" << std::endl;
	os << "    if(id < 32) {\\" << std::endl;
	os << "        // only do this for existing neurons\\" << std::endl;
	os << "        if(id < 7) {\\" << std::endl;
	os << "            if(id == 0) {\\" << std::endl;
	os << "                glbSpkCntNeurons[0] = 0;\\" << std::endl;
	os << "            }\\" << std::endl;
	os << "            glbSpkNeurons[id] = 0;\\" << std::endl;
	os << "            VNeurons[id] = (-6.50000000000000000e+01f);\\" << std::endl;
	os << "            UNeurons[id] = (-2.00000000000000000e+01f);\\" << std::endl;
	os << "            // current source variables\\" << std::endl;
	os << "        }\\" << std::endl;
	os << "    }\\" << std::endl;
	os << "\";" << std::endl;

	// *****************************************************************************
	// **********************************************************************************


}
//--------------------------------------------------------------------------
void Backend::genRunnerPreamble(CodeStream& os, const ModelSpecMerged&) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genRunnerPreamble");
}
//--------------------------------------------------------------------------
void Backend::genAllocateMemPreamble(CodeStream& os, const ModelSpecMerged& modelMerged) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genAllocateMemPreamble");
}
//--------------------------------------------------------------------------
void Backend::genStepTimeFinalisePreamble(CodeStream& os, const ModelSpecMerged& modelMerged) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genStepTimeFinalisePreamble");
}
//--------------------------------------------------------------------------
void Backend::genVariableDefinition(CodeStream& definitions, CodeStream& definitionsInternal, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\n\nBackend::genVariableDefinition called\n\n");
	const bool deviceType = isDeviceType(type);

	if (::Utils::isTypePointer(type)) {
		// Export pointer, either in definitionsInternal if variable has a device type
		// or to definitions if it should be accessable on host
		CodeStream& d = deviceType ? definitionsInternal : definitions;
		d << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
	}
	else {
		if (loc & VarLocation::HOST) {
			if (deviceType) {
				throw std::runtime_error("Variable '" + name + "' is of device-only type '" + type + "' but is located on the host");
			}

			definitions << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
		}
		if (loc & VarLocation::DEVICE) {
			// If the type is a pointer type we need a device pointer
			if (::Utils::isTypePointer(type)) {
				// Write host definition to internal definitions stream if type is device only
				CodeStream& d = deviceType ? definitionsInternal : definitions;
				d << "EXPORT_VAR " << type << " d_" << name << ";" << std::endl;
			}
			// Otherwise we just need a device variable, made volatile for safety
			else {
				definitionsInternal << "EXPORT_VAR cl::Buffer " << " db_" << name << ";" << std::endl;
			}
		}
	}
}
//--------------------------------------------------------------------------
void Backend::genVariableImplementation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genVariableImplementation");
}
//--------------------------------------------------------------------------
MemAlloc Backend::genVariableAllocation(CodeStream& os, const std::string& type, const std::string& name, VarLocation loc, size_t count) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genVariableAllocation");
	return MemAlloc::zero();
}
//--------------------------------------------------------------------------
void Backend::genVariableFree(CodeStream& os, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genVariableFree");
}
//--------------------------------------------------------------------------
void Backend::genExtraGlobalParamDefinition(CodeStream& definitions, const std::string& type, const std::string& name, VarLocation loc) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genExtraGlobalParamDefinition");
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
void Backend::genMergedGroupImplementation(CodeStream& os, const std::string& suffix, size_t idx, size_t numGroups) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMergedGroupImplementation");
}
//--------------------------------------------------------------------------
void Backend::genMergedGroupPush(CodeStream& os, const std::string& suffix, size_t idx, size_t numGroups) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMergedGroupPush");
}
//--------------------------------------------------------------------------
void Backend::genMergedExtraGlobalParamPush(CodeStream& os, const std::string& suffix, size_t mergedGroupIdx, size_t groupIdx,
	const std::string& fieldName, const std::string& egpName) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genMergedExtraGlobalParamPush");
}
//--------------------------------------------------------------------------
void Backend::genPopVariableInit(CodeStream& os, const Substitutions& kernelSubs, Handler handler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genPopVariableInit");
}
//--------------------------------------------------------------------------
void Backend::genVariableInit(CodeStream& os, const std::string&, const std::string& countVarName,
	const Substitutions& kernelSubs, Handler handler) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genVariableInit");
}
//--------------------------------------------------------------------------
void Backend::genSynapseVariableRowInit(CodeStream& os, const SynapseGroupMerged&,
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
MemAlloc Backend::genGlobalRNG(CodeStream&, CodeStream& definitionsInternal, CodeStream& runner, CodeStream&, CodeStream&) const
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
void Backend::genReturnFreeDeviceMemoryBytes(CodeStream& os) const
{
	printf("\nTO BE IMPLEMENTED: ~virtual~ CodeGenerator::OpenCL::Backend::genReturnFreeDeviceMemoryBytes");
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
bool Backend::isGlobalRNGRequired(const ModelSpecMerged& modelMerged) const
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