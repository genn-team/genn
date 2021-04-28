#pragma once

// Standard C++ includes
#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <numeric>
#include <string>

// OpenCL includes
#include "../../../../share/genn/backends/opencl/cl2.hpp"

// GeNN includes
#include "backendExport.h"

// GeNN code generator includes
#include "code_generator/backendSIMT.h"
#include "code_generator/codeStream.h"
#include "code_generator/modelSpecMerged.h"
#include "code_generator/substitutions.h"

// Forward declarations
namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::DeviceSelectMethod
//--------------------------------------------------------------------------
namespace CodeGenerator
{
namespace OpenCL
{
//! Methods for selecting OpenCL platform
enum class PlatformSelect
{
    MANUAL,         //!< Use platform specified by user
};

//! Methods for selecting OpenCL device
enum class DeviceSelect
{
    MOST_MEMORY,    //!< Pick device with most global memory
    MANUAL,         //!< Use device specified by user
};

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::WorkGroupSizeSelect
//--------------------------------------------------------------------------
//! Methods for selecting OpenCL kernel workgroup size
enum class WorkGroupSizeSelect
{
    MANUAL,     //!< Use workgroup sizes specified by user
};

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::Preferences
//--------------------------------------------------------------------------
//! Preferences for OpenCL backend
struct Preferences : public PreferencesBase
{
    Preferences()
    {
        std::fill(manualWorkGroupSizes.begin(), manualWorkGroupSizes.end(), 32);
    }

    //! How to select OpenCL platform
    PlatformSelect platformSelectMethod = PlatformSelect::MANUAL;

    //! If platform select method is set to PlatformSelect::MANUAL, id of platform to use
    unsigned int manualPlatformID = 0;

    //! How to select OpenCL device
    DeviceSelect deviceSelectMethod = DeviceSelect::MOST_MEMORY;

    //! If device select method is set to DeviceSelect::MANUAL, id of device to use
    unsigned int manualDeviceID = 0;

    //! How to select OpenCL workgroup size
    WorkGroupSizeSelect workGroupSizeSelectMethod = WorkGroupSizeSelect::MANUAL;

    //! If block size select method is set to BlockSizeSelect::MANUAL, block size to use for each kernel
    KernelBlockSize manualWorkGroupSizes;
};

//--------------------------------------------------------------------------
// CodeGenerator::OpenCL::Backend
//--------------------------------------------------------------------------
class BACKEND_EXPORT Backend : public BackendSIMT
{
public:
    Backend(const KernelBlockSize &kernelWorkGroupSizes, const Preferences &preferences,
            const std::string &scalarType, unsigned int platformIndex, unsigned int deviceIndex);

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendSIMT virtuals
    //--------------------------------------------------------------------------
    //! On some older devices, shared memory atomics are actually slower than global memory atomics so should be avoided
    virtual bool areSharedMemAtomicsSlow() const override;

    //! Get the prefix to use for shared memory variables
    virtual std::string getSharedPrefix() const override { return "__local "; }

    //! Get the ID of the current thread within the threadblock
    virtual std::string getThreadID(unsigned int axis = 0) const override{ return "get_local_id(" + std::to_string(axis) + ")"; }

    //! Get the ID of the current thread block
    virtual std::string getBlockID(unsigned int axis = 0) const override{ return "get_group_id(" + std::to_string(axis) + ")"; }

    //! Get the name of the count-leading-zeros function
    virtual std::string getCLZ() const override { return "clz"; }

    //! Get name of atomic operation
    virtual std::string getAtomic(const std::string &type, AtomicOperation op = AtomicOperation::ADD,
                                  AtomicMemSpace memSpace = AtomicMemSpace::GLOBAL) const override;

    //! Generate a shared memory barrier
    virtual void genSharedMemBarrier(CodeStream &os) const override;

    //! For SIMT backends which initialize RNGs on device, initialize population RNG with specified seed and sequence
    virtual void genPopulationRNGInit(CodeStream &os, const std::string &globalRNG, const std::string &seed, const std::string &sequence) const override;

    //! Generate a preamble to add substitution name for population RNG
    virtual void genPopulationRNGPreamble(CodeStream &os, Substitutions &subs, const std::string &globalRNG, const std::string &name = "rng") const override;

    //! If required, generate a postamble for population RNG
    /*! For example, in OpenCL, this is used to write local RNG state back to global memory*/
    virtual void genPopulationRNGPostamble(CodeStream &os, const std::string &globalRNG) const override;

    //! Generate code to skip ahead local copy of global RNG
    virtual void genGlobalRNGSkipAhead(CodeStream &os, Substitutions &subs, const std::string &sequence, const std::string &name = "rng") const override;

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendBase:: virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                                 HostHandler preambleHandler, NeuronGroupSimHandler simHandler, NeuronUpdateGroupMergedHandler wuVarUpdateHandler,
                                 HostHandler pushEGPHandler) const override;

    virtual void genSynapseUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                                  HostHandler preambleHandler, PresynapticUpdateGroupMergedHandler wumThreshHandler, PresynapticUpdateGroupMergedHandler wumSimHandler,
                                  PresynapticUpdateGroupMergedHandler wumEventHandler, PresynapticUpdateGroupMergedHandler wumProceduralConnectHandler,
                                  PostsynapticUpdateGroupMergedHandler postLearnHandler, SynapseDynamicsGroupMergedHandler synapseDynamicsHandler,
                                  HostHandler pushEGPHandler) const override;

    virtual void genCustomUpdate(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces, HostHandler preambleHandler,
                                 CustomUpdateGroupMergedHandler customUpdateHandler, CustomUpdateWUGroupMergedHandler customWUUpdateHandler,
                                 CustomUpdateTransposeWUGroupMergedHandler customWUTransposeUpdateHandler, HostHandler pushEGPHandler) const override;

    virtual void genInit(CodeStream &os, const ModelSpecMerged &modelMerged, MemorySpaces &memorySpaces,
                         HostHandler preambleHandler, NeuronInitGroupMergedHandler localNGHandler, CustomUpdateInitGroupMergedHandler cuHandler,
                         CustomWUUpdateDenseInitGroupMergedHandler cuDenseHandler, SynapseDenseInitGroupMergedHandler sgDenseInitHandler, 
                         SynapseConnectivityInitMergedGroupHandler sgSparseRowConnectHandler,  SynapseConnectivityInitMergedGroupHandler sgSparseColConnectHandler,
                         SynapseConnectivityInitMergedGroupHandler sgKernelInitHandler, SynapseSparseInitGroupMergedHandler sgSparseInitHandler, 
                         CustomWUUpdateSparseInitGroupMergedHandler cuSparseHandler, HostHandler initPushEGPHandler, HostHandler initSparsePushEGPHandler) const override;

    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genDefinitionsInternalPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &memAlloc) const override;
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const MemAlloc &allocations) const override;
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const override;

    virtual void genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count, MemAlloc &memAlloc) const override;
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const override;

    virtual void genExtraGlobalParamDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const override;
    virtual void genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name,
                                               VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;
    virtual void genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name,
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;
    virtual void genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name,
                                         VarLocation loc, const std::string &countVarName = "count", const std::string &prefix = "") const override;

    //! Generate code for pushing an updated EGP value into the merged group structure on 'device'
    virtual void genMergedExtraGlobalParamPush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx,
                                               const std::string &groupIdx, const std::string &fieldName,
                                               const std::string &egpName) const override;

    //! When generating function calls to push to merged groups, backend without equivalent of Unified Virtual Addressing e.g. OpenCL 1.2 may use different types on host
    virtual std::string getMergedGroupFieldHostType(const std::string &type) const override;

    //! When generating merged structures what type to use for simulation RNGs
    virtual std::string getMergedGroupSimRNGType() const override { return "clrngLfsr113HostStream"; }

    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const override;
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const override;

    virtual void genCurrentVariablePush(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type,
                                        const std::string &name, VarLocation loc, unsigned int batchSize) const override;
    virtual void genCurrentVariablePull(CodeStream &os, const NeuronGroupInternal &ng, const std::string &type,
                                        const std::string &name, VarLocation loc, unsigned int batchSize) const override;

    virtual void genCurrentTrueSpikePush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override
    {
        genCurrentSpikePush(os, ng, batchSize, false);
    }
    virtual void genCurrentTrueSpikePull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override
    {
        genCurrentSpikePull(os, ng, batchSize, false);
    }
    virtual void genCurrentSpikeLikeEventPush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override
    {
        genCurrentSpikePush(os, ng, batchSize, true);
    }
    virtual void genCurrentSpikeLikeEventPull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize) const override
    {
        genCurrentSpikePull(os, ng, batchSize, true);
    }

    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                                    CodeStream &allocations, CodeStream &free, MemAlloc &memAlloc) const override;
    virtual void genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations,
                                  CodeStream &free, const std::string &name, size_t count, MemAlloc &memAlloc) const override;
    virtual void genTimer(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                          CodeStream &allocations, CodeStream &free, CodeStream &stepTimeFinalise,
                          const std::string &name, bool updateInStepTime) const override;

    //! Generate code to return amount of free 'device' memory in bytes
    virtual void genReturnFreeDeviceMemoryBytes(CodeStream &os) const override;

    virtual void genMakefilePreamble(std::ostream &os) const override;
    virtual void genMakefileLinkRule(std::ostream &os) const override;
    virtual void genMakefileCompileRule(std::ostream &os) const override;

    virtual void genMSBuildConfigProperties(std::ostream &os) const override;
    virtual void genMSBuildImportProps(std::ostream &os) const override;
    virtual void genMSBuildItemDefinitions(std::ostream &os) const override;
    virtual void genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const override;
    virtual void genMSBuildImportTarget(std::ostream &os) const override;

    //! Get list of files to copy into generated code
    /*! Paths should be relative to share/genn/backends/ */
    virtual std::vector<filesystem::path> getFilesToCopy(const ModelSpecMerged &modelMerged) const override;

    //! When backends require separate 'device' and 'host' versions of variables, they are identified with a prefix.
    //! This function returns the host prefix so it can be used in otherwise platform-independent code.
    virtual std::string getHostVarPrefix() const final { return "h_"; }

    virtual std::string getPointerPrefix() const override { return "__global "; };

    //! Different backends seed RNGs in different ways. Does this one initialise population RNGS on device?
    virtual bool isPopulationRNGInitialisedOnDevice() const override { return false; }

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const override { return m_ChosenDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(); }
    
    //! Some backends will have additional small, fast, memory spaces for read-only data which might
    //! Be well-suited to storing merged group structs. This method returns the prefix required to
    //! Place arrays in these and their size in preferential order
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const override;

    virtual bool supportsNamespace() const override { return false; };

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    const cl::Device &getChosenOpenCLDevice() const { return m_ChosenDevice; }
    
    std::string getFloatAtomicAdd(const std::string &ftype, const char* memoryType = "global") const;

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    template<typename T>
    void genMergedStructPreamble(CodeStream &os, const ModelSpecMerged &modelMerged, const std::vector<T> &groups) const
    {
        // Loop through groups
        for(const auto &g : groups) {
            // Declare build kernel
            const std::string buildKernelName = "build" + T::name + std::to_string(g.getIndex()) + "Kernel";
            os << "cl::Kernel " << buildKernelName << ";" << std::endl;
            
            // Declare buffer
            os << "cl::Buffer d_merged" << T::name << "Group" << g.getIndex() << ";" << std::endl;

            // Write function to update
            os << "void pushMerged" << T::name << "Group" << g.getIndex() << "ToDevice(unsigned int idx, ";
            g.generateStructFieldArgumentDefinitions(os, *this);
            os << ")";
            {
                CodeStream::Scope b(os);

                // Add idx parameter
                os << "CHECK_OPENCL_ERRORS(" << buildKernelName << ".setArg(1, idx));" << std::endl;

                // Loop through sorted fields and add arguments
                const auto sortedFields = g.getSortedFields(*this);
                for(size_t fieldIndex = 0; fieldIndex < sortedFields.size(); fieldIndex++) {
                    const auto &f = sortedFields[fieldIndex];

                    os << "CHECK_OPENCL_ERRORS(" << buildKernelName << ".setArg(" << (2 + fieldIndex) << ", " << std::get<1>(f) << "));" << std::endl;
                }
                
                // Launch kernel
                os << "const cl::NDRange globalWorkSize(1, 1);" << std::endl;
                os << "const cl::NDRange localWorkSize(1, 1);" << std::endl;
                os << "CHECK_OPENCL_ERRORS(commandQueue.enqueueNDRangeKernel(" << buildKernelName << ", cl::NullRange, globalWorkSize, localWorkSize));" << std::endl;
            }
        }

        if(!groups.empty()) {
            // Get set of unique fields referenced in a merged group
            const auto mergedGroupFields = modelMerged.getMergedGroupFields<T>();

            // Loop through resultant fields and declare kernel for setting EGP
            for(auto f : mergedGroupFields) {
                os << "cl::Kernel setMerged" << T::name << f.mergedGroupIndex << f.fieldName << "Kernel;" << std::endl;
            }
        }
    }

    template<typename T>
    void genMergedStructBuild(CodeStream &os, const ModelSpecMerged &modelMerged, const std::vector<T> &groups, const std::string &programName) const
    {
        // Loop through groups 
        for(const auto &g : groups) {
            // Create kernel object
            const std::string kernelName = "build" + T::name + std::to_string(g.getIndex()) + "Kernel";
            os << "CHECK_OPENCL_ERRORS_POINTER(" << kernelName << " = cl::Kernel(" << programName << ", \"" << kernelName << "\", &error));" << std::endl;

            // Create group buffer
            os << "CHECK_OPENCL_ERRORS_POINTER(d_merged" << T::name << "Group" << g.getIndex() << " = cl::Buffer(clContext, CL_MEM_READ_WRITE, size_t{" << g.getStructArraySize(*this) << "}, nullptr, &error));" << std::endl;

            // Set group buffer as first kernel argument
            os << "CHECK_OPENCL_ERRORS(" << kernelName << ".setArg(0, d_merged" << T::name << "Group" << g.getIndex() << "));" << std::endl;
            os << std::endl;
        }

        if(!groups.empty()) {
            // Get set of unique fields referenced in a merged group
            const auto mergedGroupFields = modelMerged.getMergedGroupFields<T>();

            // Loop through resultant fields
            for(auto f : mergedGroupFields) {
                // Create kernel object
                const std::string kernelName = "setMerged" + T::name + std::to_string(f.mergedGroupIndex) + f.fieldName + "Kernel";
                os << "CHECK_OPENCL_ERRORS_POINTER(" << kernelName << " = cl::Kernel(" << programName << ", \"" << kernelName << "\", &error));" << std::endl;

                // Set group buffer as first kernel argument
                os << "CHECK_OPENCL_ERRORS(" << kernelName << ".setArg(0, d_merged" << T::name << "Group" << f.mergedGroupIndex << "));" << std::endl;
                os << std::endl;
            }
        }
    }

    template<typename T>
    void genMergedStructBuildKernels(CodeStream &os, const ModelSpecMerged &modelMerged, const std::vector<T> &groups) const
    {
        // Loop through groups
        for(const auto &g : groups) {
            // Generate kernel to build struct on device
            os << "__kernel void build" << T::name << g.getIndex() << "Kernel(";
            os << "__global struct Merged" << T::name << "Group" << g.getIndex() << " *group, unsigned int idx, ";

            // Loop through sorted struct fields
            const auto sortedFields = g.getSortedFields(*this);
            for(size_t fieldIndex = 0; fieldIndex < sortedFields.size(); fieldIndex++) {
                const auto &f = sortedFields[fieldIndex];
                if(::Utils::isTypePointer(std::get<0>(f))) {
                    os << "__global ";
                }
                os << std::get<0>(f) << " " << std::get<1>(f);
                if(fieldIndex != (sortedFields.size() - 1)) {
                    os << ", ";
                }
            }
            os << ")";
            {
                CodeStream::Scope b(os);

                // Assign all structure fields to values passed through parameters
                for(const auto &f : sortedFields) {
                    os << "group[idx]." << std::get<1>(f) << " = " << std::get<1>(f) << ";" << std::endl;
                }
            }
            os << std::endl;
        }

        if(!groups.empty()) {
            // Get set of unique fields referenced in a merged group
            const auto mergedGroupFields = modelMerged.getMergedGroupFields<T>();

            // Loop through resultant fields and generate push function for pointer extra global parameters
            for(auto f : mergedGroupFields) {
                
                os << "__kernel void setMerged" << T::name << f.mergedGroupIndex << f.fieldName << "Kernel(";
                os << "__global struct Merged" << T::name << "Group" << f.mergedGroupIndex << " *group, unsigned int idx, ";
                if(::Utils::isTypePointer(f.type)) {
                    os << "__global ";
                }
                os << f.type << " " << f.fieldName << ")";
                {
                    CodeStream::Scope b(os);
                    os << "group[idx]." << f.fieldName << " = " << f.fieldName << ";" << std::endl;
                }
                os << std::endl;
            }
        }
    }

    void genAtomicAddFloat(CodeStream &os, const std::string &memoryType) const;

    void genCurrentSpikePush(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize, bool spikeEvent) const
    {
        genCurrentSpikePushPull(os, ng, batchSize, spikeEvent, true);
    }

    void genCurrentSpikePull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize, bool spikeEvent) const
    {
        genCurrentSpikePushPull(os, ng, batchSize, spikeEvent, false);
    }

    void genCurrentSpikePushPull(CodeStream &os, const NeuronGroupInternal &ng, unsigned int batchSize, bool spikeEvent, bool push) const;

    void genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY = 1) const;

    void genKernelPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const;

    //! Build a string called "buildProgramFlags" containing flags to pass to cl::Program::build
    void genBuildProgramFlagsString(CodeStream &os) const;

    void divideKernelStreamInParts(CodeStream &os, const std::stringstream &kernelCode, size_t partLength) const;

    //! Tests whether chosen device is AMD    
    bool isChosenDeviceAMD() const;

    //! Tests whether chosen device is NVIDIA
    bool isChosenDeviceNVIDIA() const;

    //! Tests whether chosen platform is NVIDIA
    /*! This is primarily for Mac OS machines which have NVIDA devices on an Apple platform which,
        it would seem, doesn't support some functions e.g. inline PTAX */
    bool isChosenPlatformNVIDIA() const;

    //! Should we make all allocations from sub-buffers?
    /*! This is required for correct functioning on AMD devices */
    bool shouldUseSubBufferAllocations() const;

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const unsigned int m_ChosenPlatformIndex;
    const unsigned int m_ChosenDeviceIndex;
    unsigned int m_AllocationAlignementBytes;
    cl::Device m_ChosenDevice;
    cl::Platform m_ChosenPlatform;
};
}   // OpenCL
}   // CodeGenerator
