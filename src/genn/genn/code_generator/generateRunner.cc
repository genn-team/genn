#include "code_generator/generateRunner.h"

// Standard C++ includes
#include <sstream>
#include <string>

// GeNN includes
#include "gennUtils.h"
#include "modelSpecInternal.h"

// GeNN code generator
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/teeStream.h"
#include "code_generator/backendBase.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void writeTypeRange(CodeGenerator::CodeStream &os, const std::string &precision, const std::string &prefix)
{
    using namespace CodeGenerator;

    os << "#define " << prefix << "_MIN ";
    if (precision == "float") {
        writePreciseString(os, std::numeric_limits<float>::min());
        os << "f" << std::endl;
    }
    else {
        writePreciseString(os, std::numeric_limits<double>::min());
        os << std::endl;
    }

    os << "#define " << prefix << "_MAX ";
    if (precision == "float") {
        writePreciseString(os, std::numeric_limits<float>::max());
        os << "f" << std::endl;
    }
    else {
        writePreciseString(os, std::numeric_limits<double>::max());
        os << std::endl;
    }
    os << std::endl;
}
//-------------------------------------------------------------------------
void writeSpikeMacros(CodeGenerator::CodeStream &os, const NeuronGroupInternal &ng, bool trueSpike)
{
    const bool delayRequired = trueSpike
        ? (ng.isDelayRequired() && ng.isTrueSpikeRequired())
        : ng.isDelayRequired();
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const std::string eventMacroSuffix = trueSpike ? "" : "Event";

    // convenience macros for accessing spike count
    os << "#define spike" << eventMacroSuffix << "Count_" << ng.getName() << " glbSpkCnt" << eventSuffix << ng.getName();
    if (delayRequired) {
        os << "[spkQuePtr" << ng.getName() << "]";
    }
    else {
        os << "[0]";
    }
    os << std::endl;

    // convenience macro for accessing spikes
    os << "#define spike" << eventMacroSuffix << "_" << ng.getName();
    if (delayRequired) {
        os << " (glbSpk" << eventSuffix << ng.getName() << " + (spkQuePtr" << ng.getName() << "*" << ng.getNumNeurons() << "))";
    }
    else {
        os << " glbSpk" << eventSuffix << ng.getName();
    }
    os << std::endl;

    // convenience macro for accessing delay offset
    // **NOTE** we only require one copy of this so only ever write one for true spikes
    if(trueSpike) {
        os << "#define glbSpkShift" << ng.getName() << " ";
        if (delayRequired) {
            os << "spkQuePtr" << ng.getName() << "*" << ng.getNumNeurons();
        }
        else {
            os << "0";
        }
    }

    os << std::endl << std::endl;
}
//--------------------------------------------------------------------------
bool canPushPullVar(VarLocation loc)
{
    // A variable can be pushed and pulled if it is located on both host and device
    return ((loc & VarLocation::HOST) &&
            (loc & VarLocation::DEVICE));
}
//-------------------------------------------------------------------------
bool genVarPushPullScope(CodeGenerator::CodeStream &definitionsFunc, CodeGenerator::CodeStream &runnerPushFunc, CodeGenerator::CodeStream &runnerPullFunc,
                         VarLocation loc, const std::string &description, std::function<void()> handler)
{
    // If this variable has a location that allows pushing and pulling
    if(canPushPullVar(loc)) {
        definitionsFunc << "EXPORT_FUNC void push" << description << "ToDevice(bool uninitialisedOnly = false);" << std::endl;
        definitionsFunc << "EXPORT_FUNC void pull" << description << "FromDevice();" << std::endl;

        runnerPushFunc << "void push" << description << "ToDevice(bool uninitialisedOnly)";
        runnerPullFunc << "void pull" << description << "FromDevice()";
        {
            CodeGenerator::CodeStream::Scope a(runnerPushFunc);
            CodeGenerator::CodeStream::Scope b(runnerPullFunc);

            handler();
        }
        runnerPushFunc << std::endl;
        runnerPullFunc << std::endl;

        return true;
    }
    else {
        return false;
    }
}
//-------------------------------------------------------------------------
void genVarPushPullScope(CodeGenerator::CodeStream &definitionsFunc, CodeGenerator::CodeStream &runnerPushFunc, CodeGenerator::CodeStream &runnerPullFunc,
                         VarLocation loc, const std::string &description, std::vector<std::string> &statePushPullFunction,
                         std::function<void()> handler)
{
    // Add function to vector if push pull function was actually required
    if(genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, loc, description, handler)) {
        statePushPullFunction.push_back(description);
    }
}
//-------------------------------------------------------------------------
void genStatePushPull(CodeGenerator::CodeStream &definitionsFunc, CodeGenerator::CodeStream &runnerPushFunc, CodeGenerator::CodeStream &runnerPullFunc,
                      const std::string &name, std::vector<std::string> &statePushPullFunction)
{
    definitionsFunc << "EXPORT_FUNC void push" << name << "StateToDevice(bool uninitialisedOnly = false);" << std::endl;
    definitionsFunc << "EXPORT_FUNC void pull" << name << "StateFromDevice();" << std::endl;

    runnerPushFunc << "void push" << name << "StateToDevice(bool uninitialisedOnly)";
    runnerPullFunc << "void pull" << name << "StateFromDevice()";
    {
        CodeGenerator::CodeStream::Scope a(runnerPushFunc);
        CodeGenerator::CodeStream::Scope b(runnerPullFunc);

        for(const auto &func : statePushPullFunction) {
            runnerPushFunc << "push" << func << "ToDevice(uninitialisedOnly);" << std::endl;
            runnerPullFunc << "pull" << func << "FromDevice();" << std::endl;
        }
    }
    runnerPushFunc << std::endl;
    runnerPullFunc << std::endl;
}
//-------------------------------------------------------------------------
CodeGenerator::MemAlloc genVariable(const CodeGenerator::BackendBase &backend, CodeGenerator::CodeStream &definitionsVar, CodeGenerator::CodeStream &definitionsFunc,
                                    CodeGenerator::CodeStream &definitionsInternal, CodeGenerator::CodeStream &runner,
                                    CodeGenerator::CodeStream &allocations, CodeGenerator::CodeStream &free,
                                    CodeGenerator::CodeStream &push, CodeGenerator::CodeStream &pull,
                                    const std::string &type, const std::string &name, VarLocation loc, bool autoInitialized, size_t count,
                                    std::vector<std::string> &statePushPullFunction)
{
    // Generate push and pull functions
    genVarPushPullScope(definitionsFunc, push, pull, loc, name, statePushPullFunction,
        [&]()
        {
            backend.genVariablePushPull(push, pull, type, name, loc, autoInitialized, count);
        });

    // Generate variables
    return backend.genArray(definitionsVar, definitionsInternal, runner, allocations, free,
                            type, name, loc, count);
}
//-------------------------------------------------------------------------
void genExtraGlobalParam(const CodeGenerator::BackendBase &backend, CodeGenerator::CodeStream &definitionsVar, CodeGenerator::CodeStream &definitionsFunc,
                         CodeGenerator::CodeStream &runner, CodeGenerator::CodeStream &extraGlobalParam, const std::string &type, const std::string &name, VarLocation loc)
{
    // Generate variables
    backend.genExtraGlobalParamDefinition(definitionsVar, type, name, loc);
    backend.genExtraGlobalParamImplementation(runner, type, name, loc);

    // If type is a pointer
    if(Utils::isTypePointer(type)) {
        // Write definitions for functions to allocate and free extra global param
        definitionsFunc << "EXPORT_FUNC void allocate" << name << "(unsigned int count);" << std::endl;
        definitionsFunc << "EXPORT_FUNC void free" << name << "();" << std::endl;

        // Write allocation function
        extraGlobalParam << "void allocate" << name << "(unsigned int count)";
        {
            CodeGenerator::CodeStream::Scope a(extraGlobalParam);
            backend.genExtraGlobalParamAllocation(extraGlobalParam, type, name, loc);
        }

        // Write free function
        extraGlobalParam << "void free" << name << "()";
        {
            CodeGenerator::CodeStream::Scope a(extraGlobalParam);
            backend.genVariableFree(extraGlobalParam, name, loc);
        }

        // If variable can be pushed and pulled
        if(canPushPullVar(loc)) {
            // Write definitions for push and pull functions
            definitionsFunc << "EXPORT_FUNC void push" << name << "ToDevice(unsigned int count);" << std::endl;
            definitionsFunc << "EXPORT_FUNC void pull" << name << "FromDevice(unsigned int count);" << std::endl;

            // Write push function
            extraGlobalParam << "void push" << name << "ToDevice(unsigned int count)";
            {
                CodeGenerator::CodeStream::Scope a(extraGlobalParam);
                backend.genExtraGlobalParamPush(extraGlobalParam, type, name, loc);
            }

            // Write pull function
            extraGlobalParam << "void pull" << name << "FromDevice(unsigned int count)";
            {
                CodeGenerator::CodeStream::Scope a(extraGlobalParam);
                backend.genExtraGlobalParamPull(extraGlobalParam, type, name, loc);
            }
        }

    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
CodeGenerator::MemAlloc CodeGenerator::generateRunner(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner, const ModelSpecInternal &model,
                                                      const BackendBase &backend, int localHostID)
{
    // Track memory allocations, initially starting from zero
    auto mem = MemAlloc::zero();

    // Write definitions preamble
    definitions << "#pragma once" << std::endl;

#ifdef _WIN32
    definitions << "#ifdef BUILDING_GENERATED_CODE" << std::endl;
    definitions << "#define EXPORT_VAR __declspec(dllexport) extern" << std::endl;
    definitions << "#define EXPORT_FUNC __declspec(dllexport)" << std::endl;
    definitions << "#else" << std::endl;
    definitions << "#define EXPORT_VAR __declspec(dllimport) extern" << std::endl;
    definitions << "#define EXPORT_FUNC __declspec(dllimport)" << std::endl;
    definitions << "#endif" << std::endl;
#else
    definitions << "#define EXPORT_VAR extern" << std::endl;
    definitions << "#define EXPORT_FUNC" << std::endl;
#endif
    backend.genDefinitionsPreamble(definitions);

    // Write definitions internal preamble
    definitionsInternal << "#pragma once" << std::endl;
    definitionsInternal << "#include \"definitions.h\"" << std::endl << std::endl;
    backend.genDefinitionsInternalPreamble(definitionsInternal);
    
    // write DT macro
    if (model.getTimePrecision() == "float") {
        definitions << "#define DT " << std::to_string(model.getDT()) << "f" << std::endl;
    } else {
        definitions << "#define DT " << std::to_string(model.getDT()) << std::endl;
    }

    // Typedefine scalar type
    definitions << "typedef " << model.getPrecision() << " scalar;" << std::endl;

    // Write ranges of scalar and time types
    writeTypeRange(definitions, model.getPrecision(), "SCALAR");
    writeTypeRange(definitions, model.getTimePrecision(), "TIME");

    definitions << "// ------------------------------------------------------------------------" << std::endl;
    definitions << "// bit tool macros" << std::endl;
    definitions << "#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x" << std::endl;
    definitions << "#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1" << std::endl;
    definitions << "#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0" << std::endl;
    definitions << std::endl;

    // Write runner preamble
    runner << "#include \"definitionsInternal.h\"" << std::endl << std::endl;
    backend.genRunnerPreamble(runner);

    // Create codestreams to generate different sections of runner and definitions
    std::stringstream runnerVarDeclStream;
    std::stringstream runnerVarAllocStream;
    std::stringstream runnerVarFreeStream;
    std::stringstream runnerExtraGlobalParamFuncStream;
    std::stringstream runnerPushFuncStream;
    std::stringstream runnerPullFuncStream;
    std::stringstream runnerStepTimeFinaliseStream;
    std::stringstream definitionsVarStream;
    std::stringstream definitionsFuncStream;
    CodeStream runnerVarDecl(runnerVarDeclStream);
    CodeStream runnerVarAlloc(runnerVarAllocStream);
    CodeStream runnerVarFree(runnerVarFreeStream);
    CodeStream runnerExtraGlobalParamFunc(runnerExtraGlobalParamFuncStream);
    CodeStream runnerPushFunc(runnerPushFuncStream);
    CodeStream runnerPullFunc(runnerPullFuncStream);
    CodeStream runnerStepTimeFinalise(runnerStepTimeFinaliseStream);
    CodeStream definitionsVar(definitionsVarStream);
    CodeStream definitionsFunc(definitionsFuncStream);

    // Create a teestream to allow simultaneous writing to all streams
    TeeStream allVarStreams(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree);

    // Begin extern C block around variable declarations
    runnerVarDecl << "extern \"C\" {" << std::endl;
    definitionsVar << "extern \"C\" {" << std::endl;
    definitionsInternal << "extern \"C\" {" << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// global variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;

    // Define and declare time variables
    definitionsVar << "EXPORT_VAR unsigned long long iT;" << std::endl;
    definitionsVar << "EXPORT_VAR " << model.getTimePrecision() << " t;" << std::endl;
    runnerVarDecl << "unsigned long long iT;" << std::endl;
    runnerVarDecl << model.getTimePrecision() << " t;" << std::endl;

    // If backend requires a global RNG to simulate (or initialize) this model
    if(backend.isGlobalRNGRequired(model)) {
        mem += backend.genGlobalRNG(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree, model);
    }
    allVarStreams << std::endl;

    // Generate preamble for the final stage of time step
    // **NOTE** this is done now as there can be timing logic here
    backend.genStepTimeFinalisePreamble(runnerStepTimeFinalise, model);

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// timers" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;

    // Generate scalars to store total elapsed time
    // **NOTE** we ALWAYS generate these so usercode doesn't require #ifdefs around timing code
    backend.genScalar(definitionsVar, definitionsInternal, runnerVarDecl, "double", "neuronUpdateTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternal, runnerVarDecl, "double", "initTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternal, runnerVarDecl, "double", "presynapticUpdateTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternal, runnerVarDecl, "double", "postsynapticUpdateTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternal, runnerVarDecl, "double", "synapseDynamicsTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternal, runnerVarDecl, "double", "initSparseTime", VarLocation::HOST);

    // If timing is actually enabled
    if(model.isTimingEnabled()) {
        // Create neuron timer
        backend.genTimer(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         runnerStepTimeFinalise, "neuronUpdate", true);

        // Create init timer
        backend.genTimer(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         runnerStepTimeFinalise, "init", false);

        // If there's any synapse groups
        if(!model.getLocalSynapseGroups().empty()) {
            // Add presynaptic update timer
            backend.genTimer(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "presynapticUpdate", true);

            // Add sparse initialisation timer
            // **NOTE** this may not be required but it's not super-important
            backend.genTimer(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "initSparse", false);

            // If any synapse groups have weight update models with postsynaptic learning, add a timer
            if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                           [](const ModelSpec::SynapseGroupValueType &s){ return !s.second.getWUModel()->getLearnPostCode().empty(); }))
            {
                backend.genTimer(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 runnerStepTimeFinalise, "postsynapticUpdate", true);
            }

            // If any synapse groups have weight update models with synapse dynamics, add a timer
            if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                           [](const ModelSpec::SynapseGroupValueType &s){ return !s.second.getWUModel()->getSynapseDynamicsCode().empty(); }))
            {
                backend.genTimer(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 runnerStepTimeFinalise, "synapseDynamics", true);
            }
        }
        allVarStreams << std::endl;
    }

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// remote neuron groups" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    std::vector<std::string> currentSpikePullFunctions;
    std::vector<std::string> currentSpikeEventPullFunctions;
    for(const auto &n : model.getRemoteNeuronGroups()) {
        // Write macro so whether a neuron group is remote or not can be determined at compile time
        // **NOTE** we do this for REMOTE groups so #ifdef GROUP_NAME_REMOTE is backward compatible
        definitionsVar << "#define " << n.first << "_REMOTE" << std::endl;

        // Write convenience macros to access spikes
        writeSpikeMacros(definitionsVar, n.second, true);

        // If this neuron group has outputs to local host
        if(n.second.hasOutputToHost(localHostID)) {
            // Check that, whatever variable mode is set for these variables,
            // they are instantiated on host so they can be copied using MPI
            if(!(n.second.getSpikeLocation() & VarLocation::HOST)) {
                throw std::runtime_error("Remote neuron group '" + n.first + "' has its spike variable mode set so it is not instantiated on the host - this is not supported");
            }

            // True spike variables
            const size_t numSpikeCounts = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
            const size_t numSpikes = n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
            mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeLocation(), numSpikeCounts);
            mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "glbSpk" + n.first, n.second.getSpikeLocation(), numSpikes);

            // True spike variable push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeLocation(),
                                n.first + "CurrentSpikes", currentSpikePullFunctions,
                [&]()
                {
                    backend.genCurrentTrueSpikePush(runnerPushFunc, n.second);
                    backend.genCurrentTrueSpikePull(runnerPullFunc, n.second);
                });
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// local neuron groups" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        // Write convenience macros to access spikes
        writeSpikeMacros(definitionsVar, n.second, true);

        // True spike variables
        const size_t numSpikeCounts = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
        const size_t numSpikes = n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
        mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeLocation(), numSpikeCounts);
        mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                "unsigned int", "glbSpk" + n.first, n.second.getSpikeLocation(), numSpikes);

        // True spike push and pull functions
        genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeLocation(), n.first + "Spikes",
            [&]()
            {
                backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                            "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeLocation(), true, numSpikeCounts);
                backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                            "unsigned int", "glbSpk" + n.first, n.second.getSpikeLocation(), true, numSpikes);
            });
        
        // Current true spike push and pull functions
        genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeLocation(),
                            n.first + "CurrentSpikes", currentSpikePullFunctions,
            [&]()
            {
                backend.genCurrentTrueSpikePush(runnerPushFunc, n.second);
                backend.genCurrentTrueSpikePull(runnerPullFunc, n.second);
            });

        // If neuron ngroup eeds to emit spike-like events
        if (n.second.isSpikeEventRequired()) {
            // Write convenience macros to access spike-like events
            writeSpikeMacros(definitionsVar, n.second, false);

            // Spike-like event variables
            mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "glbSpkCntEvnt" + n.first, n.second.getSpikeEventLocation(),
                                    n.second.getNumDelaySlots());
            mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "glbSpkEvnt" + n.first, n.second.getSpikeEventLocation(),
                                    n.second.getNumNeurons() * n.second.getNumDelaySlots());

            // Spike-like event push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeEventLocation(), n.first + "SpikeEvents",
                [&]()
                {
                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                                "unsigned int", "glbSpkCntEvnt" + n.first, n.second.getSpikeLocation(), true, n.second.getNumDelaySlots());
                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                                "unsigned int", "glbSpkEvnt" + n.first, n.second.getSpikeLocation(), true, n.second.getNumNeurons() * n.second.getNumDelaySlots());
                });

            // Current spike-like event push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeEventLocation(),
                                n.first + "CurrentSpikeEvents", currentSpikeEventPullFunctions,
                [&]()
                {
                    backend.genCurrentSpikeLikeEventPush(runnerPushFunc, n.second);
                    backend.genCurrentSpikeLikeEventPull(runnerPullFunc, n.second);
                });
        }

        // If neuron group has axonal delays
        if (n.second.isDelayRequired()) {
            backend.genScalar(definitionsVar, definitionsInternal, runnerVarDecl, "unsigned int", "spkQuePtr" + n.first, VarLocation::HOST_DEVICE);
        }

        // If neuron group needs to record its spike times
        if (n.second.isSpikeTimeRequired()) {
            mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    model.getTimePrecision(), "sT" + n.first, n.second.getSpikeTimeLocation(),
                                    n.second.getNumNeurons() * n.second.getNumDelaySlots());

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeTimeLocation(), n.first + "SpikeTimes",
                [&]()
                {
                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, model.getTimePrecision(),
                                                "sT" + n.first, n.second.getSpikeTimeLocation(), true, n.second.getNumNeurons() * n.second.getNumDelaySlots());
                });
        }

        // If neuron group needs per-neuron RNGs
        if(n.second.isSimRNGRequired()) {
            mem += backend.genPopulationRNG(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree, "rng" + n.first, n.second.getNumNeurons());
        }

        // Neuron state variables
        const auto neuronModel = n.second.getNeuronModel();
        const auto vars = neuronModel->getVars();
        std::vector<std::string> neuronStatePushPullFunctions;
        for(size_t i = 0; i < vars.size(); i++) {
            const size_t count = n.second.isVarQueueRequired(i) ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
            const bool autoInitialized = !n.second.getVarInitialisers()[i].getSnippet()->getCode().empty();
            mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                               runnerPushFunc, runnerPullFunc, vars[i].type, vars[i].name + n.first,
                               n.second.getVarLocation(i), autoInitialized, count, neuronStatePushPullFunctions);
        }

        // Add helper function to push and pull entire neuron state
        genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, n.first, neuronStatePushPullFunctions);

        const auto extraGlobalParams = neuronModel->getExtraGlobalParams();
        for(size_t i = 0; i < extraGlobalParams.size(); i++) {
            genExtraGlobalParam(backend, definitionsVar, definitionsFunc, runnerVarDecl, runnerExtraGlobalParamFunc,
                                extraGlobalParams[i].type, extraGlobalParams[i].name + n.first, n.second.getExtraGlobalParamLocation(i));
        }

        if(!n.second.getCurrentSources().empty()) {
            allVarStreams << "// current source variables" << std::endl;
        }
        for (auto const *cs : n.second.getCurrentSources()) {
            const auto csModel = cs->getCurrentSourceModel();
            const auto csVars = csModel->getVars();

            std::vector<std::string> currentSourceStatePushPullFunctions;
            for(size_t i = 0; i < csVars.size(); i++) {
                const bool autoInitialized = !cs->getVarInitialisers()[i].getSnippet()->getCode().empty();
                mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                   runnerPushFunc, runnerPullFunc, csVars[i].type, csVars[i].name + cs->getName(),
                                   cs->getVarLocation(i), autoInitialized, n.second.getNumNeurons(), currentSourceStatePushPullFunctions);
            }

            // Add helper function to push and pull entire current source state
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, cs->getName(), currentSourceStatePushPullFunctions);

            const auto csExtraGlobalParams = csModel->getExtraGlobalParams();
            for(size_t i = 0; i < csExtraGlobalParams.size(); i++) {
                genExtraGlobalParam(backend, definitionsVar, definitionsFunc, runnerVarDecl, runnerExtraGlobalParamFunc,
                                    csExtraGlobalParams[i].type, csExtraGlobalParams[i].name + cs->getName(), cs->getExtraGlobalParamLocation(i));
            }
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// postsynaptic variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        // Loop through merged incoming synaptic populations
        // **NOTE** because of merging we need to loop through postsynaptic models in this
        for(const auto &m : n.second.getMergedInSyn()) {
            const auto *sg = m.first;

            mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    model.getPrecision(), "inSyn" + sg->getPSModelTargetName(), sg->getInSynLocation(),
                                    sg->getTrgNeuronGroup()->getNumNeurons());

            if (sg->isDendriticDelayRequired()) {
                mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                        model.getPrecision(), "denDelay" + sg->getPSModelTargetName(), sg->getDendriticDelayLocation(),
                                        sg->getMaxDendriticDelayTimesteps() * sg->getTrgNeuronGroup()->getNumNeurons());
                backend.genScalar(definitionsVar, definitionsInternal, runnerVarDecl, "unsigned int", "denDelayPtr" + sg->getPSModelTargetName(), VarLocation::HOST_DEVICE);
            }

            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                for(const auto &v : sg->getPSModel()->getVars()) {
                    mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                            v.type, v.name + sg->getPSModelTargetName(), sg->getPSVarLocation(v.name),
                                            sg->getTrgNeuronGroup()->getNumNeurons());
                }
            }
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// synapse connectivity" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    std::vector<std::string> connectivityPushPullFunctions;
    for(const auto &s : model.getLocalSynapseGroups()) {
        const bool autoInitialized = !s.second.getConnectivityInitialiser().getSnippet()->getRowBuildCode().empty();

        if (s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t gpSize = ((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * (size_t)s.second.getTrgNeuronGroup()->getNumNeurons()) / 32 + 1;
            mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                               runnerPushFunc, runnerPullFunc, "uint32_t", "gp" + s.first,
                               s.second.getSparseConnectivityLocation(), autoInitialized, gpSize, connectivityPushPullFunctions);

        }
        else if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            const VarLocation varLoc = s.second.getSparseConnectivityLocation();
            const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getMaxConnections();

            // Maximum row length constant
            definitionsVar << "EXPORT_VAR const unsigned int maxRowLength" << s.first << ";" << std::endl;
            runnerVarDecl << "const unsigned int maxRowLength" << s.first << " = " << s.second.getMaxConnections() << ";" << std::endl;

            // Row lengths
            mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "rowLength" + s.first, varLoc, s.second.getSrcNeuronGroup()->getNumNeurons());

            // Target indices
            mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "ind" + s.first, varLoc, size);

            // **TODO** remap is not always required
            if(backend.isSynRemapRequired() && !s.second.getWUModel()->getSynapseDynamicsCode().empty()) {
                // Allocate synRemap
                // **THINK** this is over-allocating
                mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                        "unsigned int", "synRemap" + s.first, VarLocation::DEVICE, size + 1);
            }

            // **TODO** remap is not always required
            if(backend.isPostsynapticRemapRequired() && !s.second.getWUModel()->getLearnPostCode().empty()) {
                const size_t postSize = (size_t)s.second.getTrgNeuronGroup()->getNumNeurons() * (size_t)s.second.getMaxSourceConnections();

                // Allocate column lengths
                mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                        "unsigned int", "colLength" + s.first, VarLocation::DEVICE, s.second.getTrgNeuronGroup()->getNumNeurons());

                // Allocate remap
                mem += backend.genArray(definitionsVar, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                        "unsigned int", "remap" + s.first, VarLocation::DEVICE, postSize);
            }

            // Generate push and pull functions for sparse connectivity 
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getSparseConnectivityLocation(), s.first + "Connectivity", connectivityPushPullFunctions,
                [&]()
                {
                     // Row lengths
                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                                "unsigned int", "rowLength" + s.first, s.second.getSparseConnectivityLocation(), autoInitialized, s.second.getSrcNeuronGroup()->getNumNeurons());

                    // Target indices
                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                                "unsigned int", "ind" + s.first, s.second.getSparseConnectivityLocation(), autoInitialized, size);
                });
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// synapse variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &s : model.getLocalSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        // If weight update variables should be individual
        std::vector<std::string> synapseGroupStatePushPullFunctions;
        if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            const size_t size = (s.second.getMatrixType() & SynapseMatrixConnectivity::DENSE)
                ? s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumNeurons()
                : s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getMaxConnections();

            const auto wuVars = wu->getVars();
            for(size_t i = 0; i < wuVars.size(); i++) {
                const bool autoInitialized = !s.second.getWUVarInitialisers()[i].getSnippet()->getCode().empty();
                mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                   runnerPushFunc, runnerPullFunc, wuVars[i].type, wuVars[i].name + s.first,
                                   s.second.getWUVarLocation(i), autoInitialized, size, synapseGroupStatePushPullFunctions);
            }
        }

        // Presynaptic W.U.M. variables
        const size_t preSize = (s.second.getDelaySteps() == NO_DELAY)
                ? s.second.getSrcNeuronGroup()->getNumNeurons()
                : s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getSrcNeuronGroup()->getNumDelaySlots();
        const auto wuPreVars = wu->getPreVars();
        for(size_t i = 0; i < wuPreVars.size(); i++) {
            const bool autoInitialized = !s.second.getWUPreVarInitialisers()[i].getSnippet()->getCode().empty();
            mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                               runnerPushFunc, runnerPullFunc, wuPreVars[i].type, wuPreVars[i].name + s.first,
                               s.second.getWUPreVarLocation(i), autoInitialized, preSize, synapseGroupStatePushPullFunctions);
        }

        // Postsynaptic W.U.M. variables
        const size_t postSize = (s.second.getBackPropDelaySteps() == NO_DELAY)
                ? s.second.getTrgNeuronGroup()->getNumNeurons()
                : s.second.getTrgNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumDelaySlots();
        const auto wuPostVars = wu->getPostVars();
        for(size_t i = 0; i < wuPostVars.size(); i++) {
            const bool autoInitialized = !s.second.getWUPostVarInitialisers()[i].getSnippet()->getCode().empty();
            mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternal, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                               runnerPushFunc, runnerPullFunc, wuPostVars[i].type, wuPostVars[i].name + s.first,
                               s.second.getWUPostVarLocation(i), autoInitialized, postSize, synapseGroupStatePushPullFunctions);
        }

        // If this synapse group's postsynaptic models hasn't been merged (which makes pulling them somewhat ambiguous)
        // **NOTE** we generated initialisation and declaration code earlier - here we just generate push and pull as we want this per-synapse group
        if(!s.second.isPSModelMerged()) {
            // Add code to push and pull inSyn
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getInSynLocation(), "inSyn" + s.first, synapseGroupStatePushPullFunctions,
                [&]()
                {
                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, model.getPrecision(), "inSyn" + s.first, s.second.getInSynLocation(),
                                                true, s.second.getTrgNeuronGroup()->getNumNeurons());
                });

            // If this synapse group has individual postsynaptic model variables
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                const auto psmVars = psm->getVars();
                for(size_t i = 0; i < psmVars.size(); i++) {
                    const bool autoInitialized = !s.second.getPSVarInitialisers()[i].getSnippet()->getCode().empty();
                    genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getPSVarLocation(i), psmVars[i].name + s.first, synapseGroupStatePushPullFunctions,
                        [&]()
                        {
                            backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, psmVars[i].type, psmVars[i].name + s.first, s.second.getPSVarLocation(i),
                                                        autoInitialized, s.second.getTrgNeuronGroup()->getNumNeurons());
                        });
                }
            }
        }

        // Add helper function to push and pull entire synapse group state
        genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, s.first, synapseGroupStatePushPullFunctions);

        const auto psmExtraGlobalParams = psm->getExtraGlobalParams();
        for(size_t i = 0; i < psmExtraGlobalParams.size(); i++) {
            genExtraGlobalParam(backend, definitionsVar, definitionsFunc, runnerVarDecl, runnerExtraGlobalParamFunc,
                                psmExtraGlobalParams[i].type, psmExtraGlobalParams[i].name + s.first, s.second.getPSExtraGlobalParamLocation(i));
        }

        const auto wuExtraGlobalParams = wu->getExtraGlobalParams();
        for(size_t i = 0; i < wuExtraGlobalParams.size(); i++) {
            genExtraGlobalParam(backend, definitionsVar, definitionsFunc, runnerVarDecl, runnerExtraGlobalParamFunc,
                                wuExtraGlobalParams[i].type, wuExtraGlobalParams[i].name + s.first, s.second.getWUExtraGlobalParamLocation(i));
        }

        const auto sparseConnExtraGlobalParams = s.second.getConnectivityInitialiser().getSnippet()->getExtraGlobalParams();
        for(size_t i = 0; i < sparseConnExtraGlobalParams.size(); i++) {
            genExtraGlobalParam(backend, definitionsVar, definitionsFunc, runnerVarDecl, runnerExtraGlobalParamFunc,
                                sparseConnExtraGlobalParams[i].type, sparseConnExtraGlobalParams[i].name + s.first,
                                s.second.getSparseConnectivityExtraGlobalParamLocation(i));
        }
    }
    allVarStreams << std::endl;

    // End extern C block around variable declarations
    runnerVarDecl << "}  // extern \"C\"" << std::endl;
 

    // Write variable declarations to runner
    runner << runnerVarDeclStream.str();

    // Write extra global parameter functions to runner
    runner << "// ------------------------------------------------------------------------" << std::endl;
    runner << "// extra global params" << std::endl;
    runner << "// ------------------------------------------------------------------------" << std::endl;
    runner << runnerExtraGlobalParamFuncStream.str();
    runner << std::endl;

    // Write push function declarations to runner
    runner << "// ------------------------------------------------------------------------" << std::endl;
    runner << "// copying things to device" << std::endl;
    runner << "// ------------------------------------------------------------------------" << std::endl;
    runner << runnerPushFuncStream.str();
    runner << std::endl;

    // Write pull function declarations to runner
    runner << "// ------------------------------------------------------------------------" << std::endl;
    runner << "// copying things from device" << std::endl;
    runner << "// ------------------------------------------------------------------------" << std::endl;
    runner << runnerPullFuncStream.str();
    runner << std::endl;

    // ---------------------------------------------------------------------
    // Function for copying all state to device
    runner << "void copyStateToDevice(bool uninitialisedOnly)";
    {
        CodeStream::Scope b(runner);
         for(const auto &n : model.getLocalNeuronGroups()) {
            runner << "push" << n.first << "StateToDevice(uninitialisedOnly);" << std::endl;
        }

        for(const auto &cs : model.getLocalCurrentSources()) {
            runner << "push" << cs.first << "StateToDevice(uninitialisedOnly);" << std::endl;
        }

        for(const auto &s : model.getLocalSynapseGroups()) {
            runner << "push" << s.first << "StateToDevice(uninitialisedOnly);" << std::endl;
        }
    }
    runner << std::endl;

    // ---------------------------------------------------------------------
    // Function for copying all connectivity to device
    runner << "void copyConnectivityToDevice(bool uninitialisedOnly)";
    {
        CodeStream::Scope b(runner);
        for(const auto &func : connectivityPushPullFunctions) {
            runner << "push" << func << "ToDevice(uninitialisedOnly);" << std::endl;
        }
    }
    runner << std::endl;

    // ---------------------------------------------------------------------
    // Function for copying all state from device
    runner << "void copyStateFromDevice()";
    {
        CodeStream::Scope b(runner);
        for(const auto &n : model.getLocalNeuronGroups()) {
            runner << "pull" << n.first << "StateFromDevice();" << std::endl;
        }

        for(const auto &cs : model.getLocalCurrentSources()) {
            runner << "pull" << cs.first << "StateFromDevice();" << std::endl;
        }

        for(const auto &s : model.getLocalSynapseGroups()) {
            runner << "pull" << s.first << "StateFromDevice();" << std::endl;
        }
    }
    runner << std::endl;

    // ---------------------------------------------------------------------
    // Function for copying all current spikes from device
    runner << "void copyCurrentSpikesFromDevice()";
    {
        CodeStream::Scope b(runner);
        for(const auto &func : currentSpikePullFunctions) {
            runner << "pull" << func << "FromDevice();" << std::endl;
        }
    }
    runner << std::endl;

    // ---------------------------------------------------------------------
    // Function for copying all current spikes events from device
    runner << "void copyCurrentSpikeEventsFromDevice()";
    {
        CodeStream::Scope b(runner);
        for(const auto &func : currentSpikeEventPullFunctions) {
            runner << "pull" << func << "FromDevice();" << std::endl;
        }
    }
    runner << std::endl;

    // ---------------------------------------------------------------------
    // Function for setting the CUDA device and the host's global variables.
    // Also estimates memory usage on device ...
    runner << "void allocateMem()";
    {
        CodeStream::Scope b(runner);

        // Generate preamble -this is the first bit of generated code called by user simulations
        // so global initialisation is often performed here
        backend.genAllocateMemPreamble(runner, model);

        // Write variable allocations to runner
        runner << runnerVarAllocStream.str();
    }
    runner << std::endl;

    // ------------------------------------------------------------------------
    // Function to free all global memory structures
    runner << "void freeMem()";
    {
        CodeStream::Scope b(runner);

        // Write variable frees to runner
        runner << runnerVarFreeStream.str();
    }
    runner << std::endl;

    // ------------------------------------------------------------------------
    // Function to free all global memory structures
    runner << "void stepTime()";
    {
        CodeStream::Scope b(runner);

        // Update synaptic state
        runner << "updateSynapses(t);" << std::endl;

        // Generate code to advance host-side spike queues
        for(const auto &n : model.getRemoteNeuronGroups()) {
            if(n.second.isDelayRequired() && n.second.hasOutputToHost(localHostID)) {
                runner << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
            }
        }
        for(const auto &n : model.getLocalNeuronGroups()) {
            if (n.second.isDelayRequired()) {
                runner << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
            }
        }

        // Update neuronal state
        runner << "updateNeurons(t);" << std::endl;

        // Generate code to advance host side dendritic delay buffers
        for(const auto &n : model.getLocalNeuronGroups()) {
            // Loop through incoming synaptic populations
            for(const auto &m : n.second.getMergedInSyn()) {
                const auto *sg = m.first;
                if(sg->isDendriticDelayRequired()) {
                    runner << "denDelayPtr" << sg->getPSModelTargetName() << " = (denDelayPtr" << sg->getPSModelTargetName() << " + 1) % " << sg->getMaxDendriticDelayTimesteps() << ";" << std::endl;
                }
            }
        }
        // Advance time
        runner << "iT++;" << std::endl;
        runner << "t = iT*DT;" << std::endl;

        // Write step time finalize logic to runner
        runner << runnerStepTimeFinaliseStream.str();
    }
    runner << std::endl;

    // Write variable and function definitions to header
    definitions << definitionsVarStream.str();
    definitions << definitionsFuncStream.str();

    // ---------------------------------------------------------------------
    // Function definitions
    definitions << "// Runner functions" << std::endl;
    definitions << "EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);" << std::endl;
    definitions << "EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);" << std::endl;
    definitions << "EXPORT_FUNC void copyStateFromDevice();" << std::endl;
    definitions << "EXPORT_FUNC void copyCurrentSpikesFromDevice();" << std::endl;
    definitions << "EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();" << std::endl;
    definitions << "EXPORT_FUNC void allocateMem();" << std::endl;
    definitions << "EXPORT_FUNC void freeMem();" << std::endl;
    definitions << "EXPORT_FUNC void stepTime();" << std::endl;
    definitions << std::endl;
    definitions << "// Functions generated by backend" << std::endl;
    definitions << "EXPORT_FUNC void updateNeurons(" << model.getTimePrecision() << " t);" << std::endl;
    definitions << "EXPORT_FUNC void updateSynapses(" << model.getTimePrecision() << " t);" << std::endl;
    definitions << "EXPORT_FUNC void initialize();" << std::endl;
    definitions << "EXPORT_FUNC void initializeSparse();" << std::endl;

#ifdef MPI_ENABLE
    definitions << "// MPI functions" << std::endl;
    definitions << "EXPORT_FUNC void generateMPI();" << std::endl;
#endif

    // End extern C block around definitions
    definitions << "}  // extern \"C\"" << std::endl;
    definitionsInternal << "}  // extern \"C\"" << std::endl;

    return mem;
}
