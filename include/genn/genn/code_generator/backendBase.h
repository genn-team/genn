#pragma once

// Standard C++ includes
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// PLOG includes
#include <plog/Severity.h>

// GeNN includes
#include "codeStream.h"
#include "gennExport.h"
#include "variableMode.h"

// Forward declarations
class NeuronGroupInternal;
class ModelSpecInternal;
class SynapseGroupInternal;

namespace CodeGenerator
{
    class Substitutions;
}

//--------------------------------------------------------------------------
// CodeGenerator::PreferencesBase
//--------------------------------------------------------------------------
namespace CodeGenerator
{
//! Base class for backend preferences - can be accessed via a global in 'classic' C++ code generator
struct PreferencesBase
{
    //! Generate speed-optimized code, potentially at the expense of floating-point accuracy
    bool optimizeCode = false;

    //! Generate code with debug symbols
    bool debugCode = false;

    //! C++ compiler options to be used for building all host side code (used for unix based platforms)
    std::string userCxxFlagsGNU = "";

    //! NVCC compiler options they may want to use for all GPU code (used for unix based platforms)
    std::string userNvccFlagsGNU = "";

    //! Logging level to use for code generation
    plog::Severity logLevel = plog::info;
};

//--------------------------------------------------------------------------
// CodeGenerator::MemAlloc
//--------------------------------------------------------------------------
class MemAlloc
{
public:
    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    size_t getHostBytes() const{ return m_HostBytes; }
    size_t getDeviceBytes() const{ return m_DeviceBytes; }
    size_t getZeroCopyBytes() const{ return m_ZeroCopyBytes; }
    size_t getHostMBytes() const{ return m_HostBytes / (1024 * 1024); }
    size_t getDeviceMBytes() const{ return m_DeviceBytes / (1024 * 1024); }
    size_t getZeroCopyMBytes() const{ return m_ZeroCopyBytes / (1024 * 1024); }

    //--------------------------------------------------------------------------
    // Operators
    //--------------------------------------------------------------------------
    MemAlloc &operator+=(const MemAlloc& rhs){
        m_HostBytes += rhs.m_HostBytes;
        m_DeviceBytes += rhs.m_DeviceBytes;
        m_ZeroCopyBytes += rhs.m_ZeroCopyBytes;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Static API
    //--------------------------------------------------------------------------
    static MemAlloc zero(){ return MemAlloc(0, 0, 0); }
    static MemAlloc host(size_t hostBytes){ return MemAlloc(hostBytes, 0, 0); }
    static MemAlloc device(size_t deviceBytes){ return MemAlloc(0, deviceBytes, 0); }
    static MemAlloc zeroCopy(size_t zeroCopyBytes){ return MemAlloc(0, 0, zeroCopyBytes); }

private:
    MemAlloc(size_t hostBytes, size_t deviceBytes, size_t zeroCopyBytes)
    :   m_HostBytes(hostBytes), m_DeviceBytes(deviceBytes), m_ZeroCopyBytes(zeroCopyBytes)
    {
    }

    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    size_t m_HostBytes;
    size_t m_DeviceBytes;
    size_t m_ZeroCopyBytes;
};

//--------------------------------------------------------------------------
// CodeGenerator::BackendBase
//--------------------------------------------------------------------------
class GENN_EXPORT BackendBase
{
public:
    //--------------------------------------------------------------------------
    // Typedefines
    //--------------------------------------------------------------------------
    typedef std::function<void(CodeStream &, Substitutions&)> Handler;
    
    template<typename T>
    using GroupHandler = std::function <void(CodeStream &, const T &, Substitutions&)> ;

    //! Standard callback type which provides a CodeStream to write platform-independent code for the specified NeuronGroup to.
    typedef GroupHandler<NeuronGroupInternal> NeuronGroupHandler;

    //! Standard callback type which provides a CodeStream to write platform-independent code for the specified SynapseGroup to.
    typedef GroupHandler<SynapseGroupInternal> SynapseGroupHandler;

    //! Callback function type for generation neuron group simulation code
    /*! Provides additional callbacks to insert code to emit spikes */
    typedef std::function <void(CodeStream &, const NeuronGroupInternal &, Substitutions&,
                                NeuronGroupHandler, NeuronGroupHandler)> NeuronGroupSimHandler;
    
    BackendBase(int localHostID, const std::string &scalarType);
    virtual ~BackendBase(){}

    //--------------------------------------------------------------------------
    // Declared virtuals
    //--------------------------------------------------------------------------
    //! Generate platform-specific function to update the state of all neurons
    /*! \param os                       CodeStream to write function to
        \param model                    model to generate code for
        \param simHandler               callback to write platform-independent code to update an individual NeuronGroup
        \param wuVarUpdateHandler       callback to write platform-independent code to update pre and postsynaptic weight update model variables when neuron spikes*/
    virtual void genNeuronUpdate(CodeStream &os, const ModelSpecInternal &model, NeuronGroupSimHandler simHandler, NeuronGroupHandler wuVarUpdateHandler) const = 0;

    //! Generate platform-specific function to update the state of all synapses
    /*! \param os CodeStream to write function to
        \param model model to generate code for
        \param wumThreshHandler         callback to write platform-independent code to update an individual NeuronGroup
        \param wumSimHandler            callback to write platform-independent code to process presynaptic spikes.
                                        "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided
                                        to callback via Substitutions.
        \param wumEventHandler          callback to write platform-independent code to process presynaptic spike-like events.
                                        "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided
                                        to callback via Substitutions.
        \param postLearnHandler         callback to write platform-independent code to process postsynaptic spikes.
                                        "id_pre", "id_post" and "id_syn" variables will be provided to callback via Substitutions.
        \param synapseDynamicsHandler   callback to write platform-independent code to update time-driven synapse dynamics.
                                        "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided
                                        to callback via Substitutions.*/
    virtual void genSynapseUpdate(CodeStream &os, const ModelSpecInternal &model,
                                  SynapseGroupHandler wumThreshHandler, SynapseGroupHandler wumSimHandler, SynapseGroupHandler wumEventHandler,
                                  SynapseGroupHandler postLearnHandler, SynapseGroupHandler synapseDynamicsHandler) const = 0;

    virtual void genInit(CodeStream &os, const ModelSpecInternal &model,
                         NeuronGroupHandler localNGHandler, NeuronGroupHandler remoteNGHandler,
                         SynapseGroupHandler sgDenseInitHandler, SynapseGroupHandler sgSparseConnectHandler, 
                         SynapseGroupHandler sgSparseInitHandler) const = 0;

    //! Definitions is the usercode-facing header file for the generated code. This function generates a 'preamble' to this header file.
    /*! This will be included from a standard C++ compiler so shouldn't include any platform-specific types or headers*/
    virtual void genDefinitionsPreamble(CodeStream &os) const = 0;

    //! Definitions internal is the internal header file for the generated code. This function generates a 'preamble' to this header file.
    /*! This will only be included by the platform-specific compiler used to build this backend so can include platform-specific types or headers*/
    virtual void genDefinitionsInternalPreamble(CodeStream &os) const = 0;


    virtual void genRunnerPreamble(CodeStream &os) const = 0;

    //! Allocate memory is the first function in GeNN generated code called by usercode and it should only ever be called once.
    //! Therefore it's a good place for any global initialisation. This function generates a 'preamble' to this function.
    virtual void genAllocateMemPreamble(CodeStream &os, const ModelSpecInternal &model) const = 0;

    //! After all timestep logic is complete
    virtual void genStepTimeFinalisePreamble(CodeStream &os, const ModelSpecInternal &model) const = 0;

    virtual void genVariableDefinition(CodeStream &definitions, CodeStream &definitionsInternal, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genVariableImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual MemAlloc genVariableAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const = 0;
    virtual void genVariableFree(CodeStream &os, const std::string &name, VarLocation loc) const = 0;

    virtual void genExtraGlobalParamDefinition(CodeStream &definitions, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genExtraGlobalParamImplementation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genExtraGlobalParamAllocation(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genExtraGlobalParamPush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;
    virtual void genExtraGlobalParamPull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc) const = 0;

    virtual void genPopVariableInit(CodeStream &os, VarLocation loc, const Substitutions &kernelSubs, Handler handler) const = 0;
    virtual void genVariableInit(CodeStream &os, VarLocation loc, size_t count, const std::string &indexVarName,
                                 const Substitutions &kernelSubs, Handler handler) const = 0;
    virtual void genSynapseVariableRowInit(CodeStream &os, VarLocation loc, const SynapseGroupInternal &sg,
                                           const Substitutions &kernelSubs, Handler handler) const = 0;

    virtual void genVariablePush(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const = 0;
    virtual void genVariablePull(CodeStream &os, const std::string &type, const std::string &name, VarLocation loc, size_t count) const = 0;
    virtual void genCurrentTrueSpikePush(CodeStream &os, const NeuronGroupInternal &ng) const = 0;
    virtual void genCurrentTrueSpikePull(CodeStream &os, const NeuronGroupInternal &ng) const = 0;
    virtual void genCurrentSpikeLikeEventPush(CodeStream &os, const NeuronGroupInternal &ng) const = 0;
    virtual void genCurrentSpikeLikeEventPull(CodeStream &os, const NeuronGroupInternal &ng) const = 0;

    virtual MemAlloc genGlobalRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free, const ModelSpecInternal &model) const = 0;
    virtual MemAlloc genPopulationRNG(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                                              const std::string &name, size_t count) const = 0;
    virtual void genTimer(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                          CodeStream &stepTimeFinalise, const std::string &name, bool updateInStepTime) const = 0;

    //! This function can be used to generate a preamble for the GNU makefile used to build
    virtual void genMakefilePreamble(std::ostream &os) const = 0;

    //! The GNU make build system will populate a variable called ``$(OBJECTS)`` with a list of objects to link.
    //! This function should generate a GNU make rule to build these objects into a shared library.
    virtual void genMakefileLinkRule(std::ostream &os) const = 0;

    //! The GNU make build system uses 'pattern rules' (https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html) to build backend modules into objects.
    //! This function should generate a GNU make pattern rule capable of building each module (i.e. compiling .cc file $< into .o file $@).
    virtual void genMakefileCompileRule(std::ostream &os) const = 0;

    //! In MSBuild, 'properties' are used to configure global project settings e.g. whether the MSBuild project builds a static or dynamic library
    //! This function can be used to add additional XML properties to this section.
    /*! see https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-properties for more information. */
    virtual void genMSBuildConfigProperties(std::ostream &os) const = 0;
    virtual void genMSBuildImportProps(std::ostream &os) const = 0;

    //! In MSBuild, the 'item definitions' are used to override the default properties of 'items' such as ``<ClCompile>`` or ``<Link>``.
    //! This function should generate XML to correctly configure the 'items' required to build the generated code, taking into account ``$(Configuration)`` etc.
    /*! see https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-items#item-definitions for more information. */
    virtual void genMSBuildItemDefinitions(std::ostream &os) const = 0;
    virtual void genMSBuildCompileModule(const std::string &moduleName, std::ostream &os) const = 0;
    virtual void genMSBuildImportTarget(std::ostream &os) const = 0;

    //! When backends require separate 'device' and 'host' versions of variables, they are identified with a prefix.
    //! This function returns this prefix so it can be used in otherwise platform-independent code.
    virtual std::string getVarPrefix() const{ return ""; }

    //! Different backends use different RNGs for different things. Does this one require a global RNG for the specified model?
    virtual bool isGlobalRNGRequired(const ModelSpecInternal &model) const = 0;
    virtual bool isSynRemapRequired() const = 0;
    virtual bool isPostsynapticRemapRequired() const = 0;

    //! How many bytes of memory does 'device' have
    virtual size_t getDeviceMemoryBytes() const = 0;

    //--------------------------------------------------------------------------
    // Public API
    //--------------------------------------------------------------------------
    //! Helper function to generate matching push and pull functions for a variable
    void genVariablePushPull(CodeStream &push, CodeStream &pull,
                             const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count) const
    {
        genVariablePush(push, type, name, loc, autoInitialized, count);
        genVariablePull(pull, type, name, loc, count);
    }

    //! Helper function to generate matching definition, declaration, allocation and free code for an array
    MemAlloc genArray(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                              const std::string &type, const std::string &name, VarLocation loc, size_t count) const
    {
        genVariableDefinition(definitions, definitionsInternal, type + "*", name, loc);
        genVariableImplementation(runner, type + "*", name, loc);
        genVariableFree(free, name, loc);
        return genVariableAllocation(allocations, type, name, loc, count);
    }

    //! Helper function to generate matching definition and declaration code for a scalar variable
    void genScalar(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, const std::string &type, const std::string &name, VarLocation loc) const
    {
        genVariableDefinition(definitions, definitionsInternal, type, name, loc);
        genVariableImplementation(runner, type, name, loc);
    }

    //! Gets ID of local host backend is building code for
    int getLocalHostID() const{ return m_LocalHostID; }

protected:
    //--------------------------------------------------------------------------
    // Protected API
    //--------------------------------------------------------------------------
    void addType(const std::string &type, size_t size)
    {
        m_TypeBytes.emplace(type, size);
    }

    size_t getSize(const std::string &type) const;

private:
    //--------------------------------------------------------------------------
    // Members
    //--------------------------------------------------------------------------
    const int m_LocalHostID;

    // Size of supported types in bytes - used for estimating memory usage
    std::unordered_map<std::string, size_t> m_TypeBytes;

};
}   // namespace CodeGenerator
