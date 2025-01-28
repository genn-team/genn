#pragma once

// Standard C++ includes
#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <numeric>
#include <string>

// Standard C includes
#include <cassert>

// GeNN includes
#include "backendExport.h"

// GeNN code generator includes
#include "code_generator/backendSIMT.h"
#include "code_generator/codeStream.h"

// Forward declarations
namespace filesystem
{
    class path;
}

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::PreferencesCUDAHIP
//--------------------------------------------------------------------------
//! Preferences for CUDA and HIP backends
namespace GeNN::CodeGenerator
{
struct PreferencesCUDAHIP : public PreferencesBase
{
    //! Generate corresponding NCCL batch reductions
    bool enableNCCLReductions = false;
 
    void updateHash(boost::uuids::detail::sha1 &hash) const
    {
        // Superclass 
        PreferencesBase::updateHash(hash);

        //! Update hash with preferences
        Utils::updateHash(enableNCCLReductions, hash);
    }
};

//--------------------------------------------------------------------------
// GeNN::CodeGenerator::BackendCUDAHIP
//--------------------------------------------------------------------------
class GENN_EXPORT BackendCUDAHIP : public BackendSIMT
{
public:
    BackendCUDAHIP(const KernelBlockSize &kernelBlockSizes, const PreferencesBase &preferences,
                   const std::string &runtimePrefix, const std::string &randPrefix, const std::string &cclPrefix)
    :   BackendSIMT(kernelBlockSizes, preferences), m_RuntimePrefix(runtimePrefix), 
        m_RandPrefix(randPrefix), m_CCLPrefix(cclPrefix)
    {}

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendSIMT virtuals
    //--------------------------------------------------------------------------
    //! Get the prefix to use for shared memory variables
    virtual std::string getSharedPrefix() const final{ return "__shared__ "; }

    //! Get the ID of the current thread within the threadblock
    virtual std::string getThreadID(unsigned int axis = 0) const final;

    //! Get the ID of the current thread block
    virtual std::string getBlockID(unsigned int axis = 0) const final;


    //! Get the name of the count-leading-zeros function
    virtual std::string getCLZ() const final { return "__clz"; }

    //! Generate a warp reduction across getNumLanes lanes into lane 0
    virtual void genWarpReduction(CodeStream& os, const std::string& variable,
                                  VarAccessMode access, const Type::ResolvedType& type) const final;

    //! Generate a shared memory barrier
    virtual void genSharedMemBarrier(CodeStream &os) const final;

    //! For SIMT backends which initialize RNGs on device, initialize population RNG with specified seed and sequence
    virtual void genPopulationRNGInit(CodeStream &os, const std::string &globalRNG, const std::string &seed, const std::string &sequence) const final;

    //! Generate code to skip ahead local copy of global RNG
    virtual std::string genGlobalRNGSkipAhead(CodeStream &os, const std::string &sequence) const final;

    //! Get type of population RNG
    virtual Type::ResolvedType getPopulationRNGType() const final;

    //! Generate a preamble to add substitution name for population RNG
    virtual void buildPopulationRNGEnvironment(EnvironmentGroupMergedField<NeuronUpdateGroupMerged> &env) const final;

    //! Add $(_rng) to environment based on $(_rng_internal) field with any initialisers and destructors required
    virtual void buildPopulationRNGEnvironment(EnvironmentGroupMergedField<CustomConnectivityUpdateGroupMerged> &env) const final;

    //! Some backends can produce vectorised neuron update which use 'short vector'  
    //! types like half2 to save memory bandwidth and possibly reduce compute. 
    //! Gets the vector width to use for this neuron update group.
    virtual size_t getNeuronUpdateVectorWidth(const NeuronGroupInternal &ng, const Type::TypeContext &context) const final;

    //! Some backends can produce vectorised presynaptic update which use 'short vector'  
    //! types like half2 to save memory bandwidth and possibly reduce compute. 
    //! Gets the vector width to use for this neuron update group.
    virtual size_t getPresynapticUpdateVectorWidth(const SynapseGroupInternal &sg, const Type::TypeContext &context) const final;

    //! Can this backend vectorise this variable?
    virtual bool shouldVectoriseVar(const Models::Base::Var &var, const Type::TypeContext &context) const final;
    
    //! Can this backend vectorise this variable?
    virtual bool shouldVectoriseVar(const Models::Base::CustomUpdateVar &var, const Type::TypeContext &context) const final;
    
    //! Can this backend vectorise this variable?
    virtual bool shouldVectoriseVar(const Models::Base::VarRef &, const Type::TypeContext &context) const final;

    //! Get name of short vector type used to store vectors of this sort
    virtual std::string getVectorTypeName(const Type::ResolvedType &storageType, size_t vectorWidth) const final;

    //! Get function to extract value from vector 
    virtual std::string getExtractVector(const Type::ResolvedType &type, const Type::ResolvedType &storageType, 
                                         size_t vectorWidth, size_t lane, const std::string &value) const final;

    virtual std::string getRecombineVector(const Type::ResolvedType &type, const Type::ResolvedType &storageType, 
                                           size_t vectorWidth, const std::string &valuePrefix) const final;

    //--------------------------------------------------------------------------
    // CodeGenerator::BackendBase virtuals
    //--------------------------------------------------------------------------
    virtual void genNeuronUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                 HostHandler preambleHandler) const final;

    virtual void genSynapseUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                  HostHandler preambleHandler) const final;

    virtual void genCustomUpdate(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                                 HostHandler preambleHandler) const final;

    virtual void genInit(CodeStream &os, ModelSpecMerged &modelMerged, BackendBase::MemorySpaces &memorySpaces, 
                         HostHandler preambleHandler) const final;

    virtual void genDefinitionsPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genRunnerPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genFreeMemPreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecMerged &modelMerged) const final;

    //! Create array of backend-specific population RNGs (if they are initialised on host this will occur here)
    /*! \param count        number of RNGs required*/
    virtual std::unique_ptr<GeNN::Runtime::ArrayBase> createPopulationRNG(size_t count) const final;

    //! Generate code for pushing a variable with a size known at runtime to the 'device'
    virtual void genLazyVariableDynamicPush(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName) const final;

    //! Generate code for pulling a variable with a size known at runtime from the 'device'
    virtual void genLazyVariableDynamicPull(CodeStream &os, 
                                            const Type::ResolvedType &type, const std::string &name,
                                            VarLocation loc, const std::string &countVarName) const final;

    //! Generate code for pushing a new pointer to a dynamic variable into the merged group structure on 'device'
    virtual void genMergedDynamicVariablePush(CodeStream &os, const std::string &suffix, size_t mergedGroupIdx, 
                                              const std::string &groupIdx, const std::string &fieldName,
                                              const std::string &egpName) const final;

    //! When generating function calls to push to merged groups, backend without equivalent of Unified Virtual Addressing e.g. OpenCL 1.2 may use different types on host
    virtual std::string getMergedGroupFieldHostTypeName(const Type::ResolvedType &type) const final;

    //! Get function to convert value from storageType to type
    virtual std::string getStorageToTypeConversion(const Type::ResolvedType &type, const Type::ResolvedType &storageType, const std::string &value) const final;

    //! Get function to convert value from type to storageType
    virtual std::string getTypeToStorageConversion(const Type::ResolvedType &type, const Type::ResolvedType &storageType, const std::string &value) const final;
    
    //! Generate a single RNG instance
    /*! On single-threaded platforms this can be a standard RNG like M.T. but, on parallel platforms, it is likely to be a counter-based RNG */
    virtual void genGlobalDeviceRNG(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free) const final;

    virtual void genTimer(CodeStream &definitions, CodeStream &runner, CodeStream &allocations, CodeStream &free, 
                          CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const final;

    //! Generate code to return amount of free 'device' memory in bytes
    virtual void genReturnFreeDeviceMemoryBytes(CodeStream &os) const final;

    // Is this combination of type and storage type permitted for variables on this backend?
    virtual bool isVarTypePermitted(const Type::ResolvedType &type, const Type::ResolvedType &storageType) const final;

    //! As well as host pointers, are device objects required?
    virtual bool isArrayDeviceObjectRequired() const final{ return true; }

    //! As well as host pointers, are additional host objects required e.g. for buffers in OpenCL?
    virtual bool isArrayHostObjectRequired() const final{ return false; }

    //! Different backends seed RNGs in different ways. Does this one initialise population RNGS on device?
    virtual bool isPopulationRNGInitialisedOnDevice() const final { return true; }

    //! Backends which support batch-parallelism might require an additional host reduction phase after reduction kernels
    virtual bool isHostReductionRequired() const final { return getPreferences<PreferencesCUDAHIP>().enableNCCLReductions; }

    //! Some backends will have additional small, fast, memory spaces for read-only data which might
    //! Be well-suited to storing merged group structs. This method returns the prefix required to
    //! Place arrays in these and their size in preferential order
    virtual MemorySpaces getMergedGroupMemorySpaces(const ModelSpecMerged &modelMerged) const final;

protected:
    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    //! Get the safe amount of constant cache we can use
    virtual size_t getChosenDeviceSafeConstMemBytes() const = 0;

    //! Get internal type population RNG gets loaded into
    virtual Type::ResolvedType getPopulationRNGInternalType() const = 0;

    //! Get library of RNG functions to use
    virtual const EnvironmentLibrary::Library &getRNGFunctions(const Type::ResolvedType &precision) const = 0;

    //! Generate HIP/CUDA specific bits of definitions preamble
    virtual void genDefinitionsPreambleInternal(CodeStream &os, const ModelSpecMerged &modelMerged) const = 0;

    virtual void genKernelDimensions(CodeStream &os, Kernel kernel, size_t numThreadsX, size_t batchSize, size_t numBlockThreadsY = 1) const = 0;

    //--------------------------------------------------------------------------
    // Protected methods
    //--------------------------------------------------------------------------
    const std::string &getRuntimePrefix() const{ return m_RuntimePrefix; }
    const std::string &getRandPrefix() const{ return m_RandPrefix; }
    const std::string &getCCLPrefix() const{ return m_CCLPrefix; }

private:
    //--------------------------------------------------------------------------
    // Private methods
    //--------------------------------------------------------------------------
    std::string getNCCLReductionType(VarAccessMode mode) const;
    std::string getNCCLType(const Type::ResolvedType &type) const;

    template<typename T>
    void genMergedStructArrayPush(CodeStream &os, const std::vector<T> &groups) const
    {
        // Loop through groups
        for(const auto &g : groups) {
            // Check that a memory space has been assigned
            assert(!g.getMemorySpace().empty());

            // Implement merged group array in previously assigned memory space
            os << g.getMemorySpace() << " Merged" << T::name << "Group" << g.getIndex() << " d_merged" << T::name << "Group" << g.getIndex() << "[" << g.getGroups().size() << "];" << std::endl;
            if(!g.getFields().empty()) {
                os << "void pushMerged" << T::name << "Group" << g.getIndex() << "ToDevice(unsigned int idx, ";
                g.generateStructFieldArgumentDefinitions(os, *this);
                os << ")";
                {
                    CodeStream::Scope b(os);

                    // Loop through sorted fields and build struct on the stack
                    os << "Merged" << T::name << "Group" << g.getIndex() << " group = {";
                    const auto sortedFields = g.getSortedFields(*this);
                    for(const auto &f : sortedFields) {
                        os << f.name << ", ";
                    }
                    os << "};" << std::endl;

                    // Push to device
                    os << "CHECK_RUNTIME_ERRORS(" << getRuntimePrefix() << "MemcpyToSymbolAsync(d_merged" << T::name << "Group" << g.getIndex() << ", &group, ";
                    os << "sizeof(Merged" << T::name << "Group" << g.getIndex() << "), idx * sizeof(Merged" << T::name << "Group" << g.getIndex() << "), ";
                    os << getRuntimePrefix() << "MemcpyHostToDevice, 0));" << std::endl;
                }
            }
        }
    }

    template<typename G>
    void genNCCLReduction(EnvironmentExternalBase &env, G &cg) const
    {
        CodeStream::Scope b(env.getStream());
        env.getStream() << "// merged custom update host reduction group " << cg.getIndex() << std::endl;
        env.getStream() << "for(unsigned int g = 0; g < " << cg.getGroups().size() << "; g++)";
        {
            CodeStream::Scope b(env.getStream());

            // Get reference to group
            env.getStream() << "const auto *group = &merged" << G::name << "Group" << cg.getIndex() << "[g]; " << std::endl;
            EnvironmentGroupMergedField<G> groupEnv(env, cg);
            buildSizeEnvironment(groupEnv);

            // Loop through variables
            const auto *cm = cg.getArchetype().getModel();
            for(const auto &v : cm->getVars()) {
                // If variable is reduction target
                if(v.access & VarAccessModeAttribute::REDUCE) {
                    // Add pointer field
                    const auto resolvedStorageType = v.storageType.resolve(cg.getTypeContext());
                    groupEnv.addField(resolvedStorageType.createPointer(), "_" + v.name, v.name,
                                      [v](const auto &runtime, const auto &g, size_t) 
                                      { 
                                          return runtime.getArray(g, v.name);
                                      });
                    
                    // Add NCCL reduction
                    groupEnv.print("CHECK_NCCL_ERRORS(ncclAllReduce($(_" + v.name + "), $(_" + v.name + "), $(_size)");
                    groupEnv.printLine(", " + getNCCLType(resolvedStorageType) + ", " + getNCCLReductionType(getVarAccessMode(v.access)) + ", ncclCommunicator, 0));");
                }
            }

            // Loop through variable references
            // **TODO** storage type
            for(const auto &v : cm->getVarRefs()) {
                // If variable reference ios reduction target
                if(v.access & VarAccessModeAttribute::REDUCE) {
                    // Add pointer field
                    const auto resolvedStorageType = v.storageType.resolve(cg.getTypeContext());
                    groupEnv.addField(resolvedStorageType.createPointer(), "_" + v.name, v.name,
                                      [v](const auto &runtime, const auto &g, size_t) 
                                      { 
                                          const auto varRef = g.getVarReferences().at(v.name);
                                          return varRef.getTargetArray(runtime);
                                      });

                    // Add NCCL reduction
                    groupEnv.print("CHECK_NCCL_ERRORS(ncclAllReduce($(_" + v.name + "), $(_" + v.name + "), $(_size)");
                    groupEnv.printLine(", " + getNCCLType(resolvedStorageType) + ", " + getNCCLReductionType(v.access) + ", ncclCommunicator, 0));");
                }
            } 
        }
    }
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    std::string m_RuntimePrefix;
    std::string m_RandPrefix;
    std::string m_CCLPrefix;
};
}   // GeNN::CodeGenerator
