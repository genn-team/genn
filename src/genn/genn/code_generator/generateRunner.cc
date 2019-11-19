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
template<typename T>
class MergedStructGenerator
{
public:
    typedef std::function<std::string(const typename T::GroupInternal &)> GetFieldValueFunc;

    void addField(const std::string &name, GetFieldValueFunc getFieldValue)
    {
        m_Fields.emplace_back(name, getFieldValue);
    }

    void addVars(const std::vector<Models::Base::Var> &vars, const std::string &prefix)
    {
        for(const auto &v : vars) {
            addField(v.type + "* " + v.name,
                     [v, prefix](const typename T::GroupInternal &g){ return prefix + v.name + g.getName(); });
        }
    }

    void addEGPs(const std::vector<Snippet::Base::EGP> &egps)
    {
        for(const auto &e : egps) {
            addField(e.type + " " + e.name,
                     [e](const typename T::GroupInternal &g){ return e.name + g.getName(); });
        }
    }

    void generate(CodeGenerator::CodeStream &definitionsInternal, CodeGenerator::CodeStream &definitionsInternalFunc,
                  CodeGenerator::CodeStream &runnerVarAlloc, const std::string &name, const T &m)
    {
        // Write struct declation to top of definitions internal
        definitionsInternal << "struct Merged" << name << "Group" << m.getIndex() << std::endl;
        {
            CodeGenerator::CodeStream::Scope b(definitionsInternal);
            for(const auto &f : m_Fields) {
                definitionsInternal << f.first << ";" << std::endl;
            }
            definitionsInternal << std::endl;
        }

        definitionsInternal << ";" << std::endl;

        // Write local array of these structs containing individual neuron group pointers etc
        runnerVarAlloc << "Merged" << name << "Group" << m.getIndex() << " merged" << name <<  "Group" << m.getIndex() << "[] = ";
        {
            CodeGenerator::CodeStream::Scope b(runnerVarAlloc);
            for(const auto &sg : m.getGroups()) {
                runnerVarAlloc << "{";
                for(const auto &f : m_Fields) {
                    runnerVarAlloc << f.second(sg) << ", ";
                }
                runnerVarAlloc << "}," << std::endl;
            }
        }
        runnerVarAlloc << ";" << std::endl;

        // Then generate call to function to copy local array to device
        runnerVarAlloc << "pushMerged" << name << "Group" << m.getIndex() << "ToDevice(merged" << name << "Group" << m.getIndex() << ");" << std::endl;

        // Finally add declaration to function to definitions internal
        definitionsInternalFunc << "EXPORT_FUNC void pushMerged" << name << "Group" << m.getIndex() << "ToDevice(const Merged" << name << "Group" << m.getIndex() << " *group);" << std::endl;
    }
private:
    std::vector<std::pair<std::string, GetFieldValueFunc>> m_Fields;
};

void genTypeRange(CodeGenerator::CodeStream &os, const std::string &precision, const std::string &prefix)
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
void genSpikeMacros(CodeGenerator::CodeStream &os, const NeuronGroupInternal &ng, bool trueSpike)
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
        os << " (glbSpk" << eventSuffix << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "))";
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
void genVarGetterScope(CodeGenerator::CodeStream &definitionsFunc, CodeGenerator::CodeStream &runnerGetterFunc,
                       VarLocation loc, const std::string &description, const std::string &type, std::function<void()> handler)
{
    // If this variable has a location that allows pushing and pulling and hence getting a host pointer
    if(canPushPullVar(loc)) {
        // Export getter
        definitionsFunc << "EXPORT_FUNC " << type << " get" << description << "();" << std::endl;

        // Define getter
        runnerGetterFunc << type << " get" << description << "()";
        {
            CodeGenerator::CodeStream::Scope a(runnerGetterFunc);
            handler();
        }
        runnerGetterFunc << std::endl;
    }
}
//-------------------------------------------------------------------------
void genSpikeGetters(CodeGenerator::CodeStream &definitionsFunc, CodeGenerator::CodeStream &runnerGetterFunc,
                     const NeuronGroupInternal &ng, bool trueSpike)
{
    const std::string eventSuffix = trueSpike ? "" : "Evnt";
    const bool delayRequired = trueSpike
        ? (ng.isDelayRequired() && ng.isTrueSpikeRequired())
        : ng.isDelayRequired();
    const VarLocation loc = trueSpike ? ng.getSpikeLocation() : ng.getSpikeEventLocation();

    // Generate getter for current spike counts
    genVarGetterScope(definitionsFunc, runnerGetterFunc,
                      loc, ng.getName() +  (trueSpike ? "CurrentSpikes" : "CurrentSpikeEvents"), "unsigned int*",
                      [&]()
                      {
                          runnerGetterFunc << "return ";
                          if (delayRequired) {
                              runnerGetterFunc << " (glbSpk" << eventSuffix << ng.getName() << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << "));";
                          }
                          else {
                              runnerGetterFunc << " glbSpk" << eventSuffix << ng.getName() << ";";
                          }
                          runnerGetterFunc << std::endl;
                      });

    // Generate getter for current spikes
    genVarGetterScope(definitionsFunc, runnerGetterFunc,
                      loc, ng.getName() + (trueSpike ? "CurrentSpikeCount" : "CurrentSpikeEventCount"), "unsigned int&",
                      [&]()
                      {
                          runnerGetterFunc << "return glbSpkCnt" << eventSuffix << ng.getName();
                          if (delayRequired) {
                              runnerGetterFunc << "[spkQuePtr" << ng.getName() << "];";
                          }
                          else {
                              runnerGetterFunc << "[0];";
                          }
                          runnerGetterFunc << std::endl;
                      });


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
CodeGenerator::MemAlloc CodeGenerator::generateRunner(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                                                      const ModelSpecInternal &model, const BackendBase &backend, int localHostID)
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
    backend.genDefinitionsPreamble(definitions, model);

    // Write definitions internal preamble
    definitionsInternal << "#pragma once" << std::endl;
    definitionsInternal << "#include \"definitions.h\"" << std::endl << std::endl;
    backend.genDefinitionsInternalPreamble(definitionsInternal, model);
    
    // write DT macro
    if (model.getTimePrecision() == "float") {
        definitions << "#define DT " << std::to_string(model.getDT()) << "f" << std::endl;
    } else {
        definitions << "#define DT " << std::to_string(model.getDT()) << std::endl;
    }

    // Typedefine scalar type
    definitions << "typedef " << model.getPrecision() << " scalar;" << std::endl;

    // Write ranges of scalar and time types
    genTypeRange(definitions, model.getPrecision(), "SCALAR");
    genTypeRange(definitions, model.getTimePrecision(), "TIME");

    definitions << "// ------------------------------------------------------------------------" << std::endl;
    definitions << "// bit tool macros" << std::endl;
    definitions << "#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x" << std::endl;
    definitions << "#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1" << std::endl;
    definitions << "#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0" << std::endl;
    definitions << std::endl;

    // Write runner preamble
    runner << "#include \"definitionsInternal.h\"" << std::endl << std::endl;
    backend.genRunnerPreamble(runner, model);

    // Create codestreams to generate different sections of runner and definitions
    std::stringstream runnerVarDeclStream;
    std::stringstream runnerVarAllocStream;
    std::stringstream runnerVarFreeStream;
    std::stringstream runnerExtraGlobalParamFuncStream;
    std::stringstream runnerPushFuncStream;
    std::stringstream runnerPullFuncStream;
    std::stringstream runnerGetterFuncStream;
    std::stringstream runnerStepTimeFinaliseStream;
    std::stringstream definitionsVarStream;
    std::stringstream definitionsFuncStream;
    std::stringstream definitionsInternalVarStream;
    std::stringstream definitionsInternalFuncStream;
    CodeStream runnerVarDecl(runnerVarDeclStream);
    CodeStream runnerVarAlloc(runnerVarAllocStream);
    CodeStream runnerVarFree(runnerVarFreeStream);
    CodeStream runnerExtraGlobalParamFunc(runnerExtraGlobalParamFuncStream);
    CodeStream runnerPushFunc(runnerPushFuncStream);
    CodeStream runnerPullFunc(runnerPullFuncStream);
    CodeStream runnerGetterFunc(runnerGetterFuncStream);
    CodeStream runnerStepTimeFinalise(runnerStepTimeFinaliseStream);
    CodeStream definitionsVar(definitionsVarStream);
    CodeStream definitionsFunc(definitionsFuncStream);
    CodeStream definitionsInternalVar(definitionsInternalVarStream);
    CodeStream definitionsInternalFunc(definitionsInternalFuncStream);

    // Create a teestream to allow simultaneous writing to all streams
    TeeStream allVarStreams(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree);

    // Begin extern C block around variable declarations
    runnerVarDecl << "extern \"C\" {" << std::endl;
    definitionsVar << "extern \"C\" {" << std::endl;
    definitionsInternalVar << "extern \"C\" {" << std::endl;

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
        mem += backend.genGlobalRNG(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree);
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
    backend.genScalar(definitionsVar, definitionsInternalVar, runnerVarDecl, "double", "neuronUpdateTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternalVar, runnerVarDecl, "double", "initTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternalVar, runnerVarDecl, "double", "presynapticUpdateTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternalVar, runnerVarDecl, "double", "postsynapticUpdateTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternalVar, runnerVarDecl, "double", "synapseDynamicsTime", VarLocation::HOST);
    backend.genScalar(definitionsVar, definitionsInternalVar, runnerVarDecl, "double", "initSparseTime", VarLocation::HOST);

    // If timing is actually enabled
    if(model.isTimingEnabled()) {
        // Create neuron timer
        backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         runnerStepTimeFinalise, "neuronUpdate", true);

        // Create init timer
        backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         runnerStepTimeFinalise, "init", false);

        // If there's any synapse groups
        if(!model.getLocalSynapseGroups().empty()) {
            // If any synapse groups process spikes or spike-like events, add a timer
            if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                           [](const ModelSpec::SynapseGroupValueType &s){ return (s.second.isSpikeEventRequired() || s.second.isTrueSpikeRequired()); }))
            {
                backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 runnerStepTimeFinalise, "presynapticUpdate", true);
            }

            // Add sparse initialisation timer
            // **FIXME** this will cause problems if no sparse initialization kernel is required
            backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "initSparse", false);

            // If any synapse groups have weight update models with postsynaptic learning, add a timer
            if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                           [](const ModelSpec::SynapseGroupValueType &s){ return !s.second.getWUModel()->getLearnPostCode().empty(); }))
            {
                backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 runnerStepTimeFinalise, "postsynapticUpdate", true);
            }

            // If any synapse groups have weight update models with synapse dynamics, add a timer
            if(std::any_of(model.getLocalSynapseGroups().cbegin(), model.getLocalSynapseGroups().cend(),
                           [](const ModelSpec::SynapseGroupValueType &s){ return !s.second.getWUModel()->getSynapseDynamicsCode().empty(); }))
            {
                backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
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
        genSpikeMacros(definitionsVar, n.second, true);

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
            mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeLocation(), numSpikeCounts);
            mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "glbSpk" + n.first, n.second.getSpikeLocation(), numSpikes);

            // True spike variable push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeLocation(),
                                n.first + "CurrentSpikes", currentSpikePullFunctions,
                [&]()
                {
                    backend.genCurrentTrueSpikePush(runnerPushFunc, n.second);
                    backend.genCurrentTrueSpikePull(runnerPullFunc, n.second);
                });

            // Current true spike getter functions
            genSpikeGetters(definitionsFunc, runnerGetterFunc, n.second, true);
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// local neuron groups" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &n : model.getLocalNeuronGroups()) {
        // Write convenience macros to access spikes
        genSpikeMacros(definitionsVar, n.second, true);

        // True spike variables
        const size_t numSpikeCounts = n.second.isTrueSpikeRequired() ? n.second.getNumDelaySlots() : 1;
        const size_t numSpikes = n.second.isTrueSpikeRequired() ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
        mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeLocation(), numSpikeCounts);
        mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
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

        // Current true spike getter functions
        genSpikeGetters(definitionsFunc, runnerGetterFunc, n.second, true);

        // If neuron ngroup eeds to emit spike-like events
        if (n.second.isSpikeEventRequired()) {
            // Write convenience macros to access spike-like events
            genSpikeMacros(definitionsVar, n.second, false);

            // Spike-like event variables
            mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "glbSpkCntEvnt" + n.first, n.second.getSpikeEventLocation(),
                                    n.second.getNumDelaySlots());
            mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
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

            // Current true spike getter functions
            genSpikeGetters(definitionsFunc, runnerGetterFunc, n.second, false);
        }

        // If neuron group has axonal delays
        if (n.second.isDelayRequired()) {
            backend.genScalar(definitionsVar, definitionsInternalVar, runnerVarDecl, "unsigned int", "spkQuePtr" + n.first, VarLocation::HOST_DEVICE);
        }

        // If neuron group needs to record its spike times
        if (n.second.isSpikeTimeRequired()) {
            mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
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
            mem += backend.genPopulationRNG(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree, "rng" + n.first, n.second.getNumNeurons());
        }

        // Neuron state variables
        const auto neuronModel = n.second.getNeuronModel();
        const auto vars = neuronModel->getVars();
        std::vector<std::string> neuronStatePushPullFunctions;
        for(size_t i = 0; i < vars.size(); i++) {
            const size_t count = n.second.isVarQueueRequired(i) ? n.second.getNumNeurons() * n.second.getNumDelaySlots() : n.second.getNumNeurons();
            const bool autoInitialized = !n.second.getVarInitialisers()[i].getSnippet()->getCode().empty();
            mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                               runnerPushFunc, runnerPullFunc, vars[i].type, vars[i].name + n.first,
                               n.second.getVarLocation(i), autoInitialized, count, neuronStatePushPullFunctions);

            // Current variable push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getVarLocation(i),
                                "Current" + vars[i].name + n.first,
                [&]()
                {
                    backend.genCurrentVariablePushPull(runnerPushFunc, runnerPullFunc, n.second, vars[i].type,
                                                       vars[i].name, n.second.getVarLocation(i));
                });

            // Write getter to get access to correct pointer
            const bool delayRequired = (n.second.isVarQueueRequired(i) &&  n.second.isDelayRequired());
            genVarGetterScope(definitionsFunc, runnerGetterFunc, n.second.getVarLocation(i),
                              "Current" + vars[i].name + n.first, vars[i].type + "*",
                [&]()
                {
                    if(delayRequired) {
                        runnerGetterFunc << "return " << vars[i].name << n.first << " + (spkQuePtr" << n.first << " * " << n.second.getNumNeurons() << ");" << std::endl;
                    }
                    else {
                        runnerGetterFunc << "return " << vars[i].name << n.first << ";" << std::endl;
                    }
                });
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
                mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
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

            mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    model.getPrecision(), "inSyn" + sg->getPSModelTargetName(), sg->getInSynLocation(),
                                    sg->getTrgNeuronGroup()->getNumNeurons());

            if (sg->isDendriticDelayRequired()) {
                mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                        model.getPrecision(), "denDelay" + sg->getPSModelTargetName(), sg->getDendriticDelayLocation(),
                                        sg->getMaxDendriticDelayTimesteps() * sg->getTrgNeuronGroup()->getNumNeurons());
                backend.genScalar(definitionsVar, definitionsInternalVar, runnerVarDecl, "unsigned int", "denDelayPtr" + sg->getPSModelTargetName(), VarLocation::HOST_DEVICE);
            }

            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                for(const auto &v : sg->getPSModel()->getVars()) {
                    mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
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
    for(const auto &sMerge : model.getMergedLocalSynapseGroups()) {
        for(const auto &s : sMerge.getGroups()) {
            const bool autoInitialized = !s.get().getConnectivityInitialiser().getSnippet()->getRowBuildCode().empty();

            if (s.get().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                const size_t gpSize = ((size_t)s.get().getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(sMerge, s.get())) / 32 + 1;
                mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                runnerPushFunc, runnerPullFunc, "uint32_t", "gp" + s.get().getName(),
                                s.get().getSparseConnectivityLocation(), autoInitialized, gpSize, connectivityPushPullFunctions);

            }
            else if(s.get().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                const VarLocation varLoc = s.get().getSparseConnectivityLocation();
                const size_t size = s.get().getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(sMerge, s.get());

                // Maximum row length constant
                definitionsVar << "EXPORT_VAR const unsigned int maxRowLength" << s.get().getName() << ";" << std::endl;
                runnerVarDecl << "const unsigned int maxRowLength" << s.get().getName() << " = " << backend.getSynapticMatrixRowStride(sMerge, s.get()) << ";" << std::endl;

                // Row lengths
                mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                        "unsigned int", "rowLength" + s.get().getName(), varLoc, s.get().getSrcNeuronGroup()->getNumNeurons());

                // Target indices
                mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                        s.get().getSparseIndType(), "ind" + s.get().getName(), varLoc, size);

                // **TODO** remap is not always required
                if(backend.isSynRemapRequired() && !s.get().getWUModel()->getSynapseDynamicsCode().empty()) {
                    // Allocate synRemap
                    // **THINK** this is over-allocating
                    mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                            "unsigned int", "synRemap" + s.get().getName(), VarLocation::DEVICE, size + 1);
                }

                // **TODO** remap is not always required
                if(backend.isPostsynapticRemapRequired() && !s.get().getWUModel()->getLearnPostCode().empty()) {
                    const size_t postSize = (size_t)s.get().getTrgNeuronGroup()->getNumNeurons() * (size_t)s.get().getMaxSourceConnections();

                    // Allocate column lengths
                    mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                            "unsigned int", "colLength" + s.get().getName(), VarLocation::DEVICE, s.get().getTrgNeuronGroup()->getNumNeurons());

                    // Allocate remap
                    mem += backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                            "unsigned int", "remap" + s.get().getName(), VarLocation::DEVICE, postSize);
                }

                // Generate push and pull functions for sparse connectivity
                genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc,
                                    s.get().getSparseConnectivityLocation(), s.get().getName() + "Connectivity", connectivityPushPullFunctions,
                    [&]()
                    {
                        // Row lengths
                        backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                                    "unsigned int", "rowLength" + s.get().getName(), s.get().getSparseConnectivityLocation(), autoInitialized, s.get().getSrcNeuronGroup()->getNumNeurons());

                        // Target indices
                        backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                                    "unsigned int", "ind" + s.get().getName(), s.get().getSparseConnectivityLocation(), autoInitialized, size);
                    });
            }
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// synapse variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &sMerge : model.getMergedLocalSynapseGroups()) {
        for(const auto &s : sMerge.getGroups()) {
            const auto *wu = s.get().getWUModel();
            const auto *psm = s.get().getPSModel();

            // If weight update variables should be individual
            std::vector<std::string> synapseGroupStatePushPullFunctions;
            if (s.get().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
                const size_t size = s.get().getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(sMerge, s.get());

                const auto wuVars = wu->getVars();
                for(size_t i = 0; i < wuVars.size(); i++) {
                    const bool autoInitialized = !s.get().getWUVarInitialisers()[i].getSnippet()->getCode().empty();
                    mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    runnerPushFunc, runnerPullFunc, wuVars[i].type, wuVars[i].name + s.get().getName(),
                                    s.get().getWUVarLocation(i), autoInitialized, size, synapseGroupStatePushPullFunctions);
                }
            }

            // Presynaptic W.U.M. variables
            const size_t preSize = (s.get().getDelaySteps() == NO_DELAY)
                    ? s.get().getSrcNeuronGroup()->getNumNeurons()
                    : s.get().getSrcNeuronGroup()->getNumNeurons() * s.get().getSrcNeuronGroup()->getNumDelaySlots();
            const auto wuPreVars = wu->getPreVars();
            for(size_t i = 0; i < wuPreVars.size(); i++) {
                const bool autoInitialized = !s.get().getWUPreVarInitialisers()[i].getSnippet()->getCode().empty();
                mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                runnerPushFunc, runnerPullFunc, wuPreVars[i].type, wuPreVars[i].name + s.get().getName(),
                                s.get().getWUPreVarLocation(i), autoInitialized, preSize, synapseGroupStatePushPullFunctions);
            }

            // Postsynaptic W.U.M. variables
            const size_t postSize = (s.get().getBackPropDelaySteps() == NO_DELAY)
                    ? s.get().getTrgNeuronGroup()->getNumNeurons()
                    : s.get().getTrgNeuronGroup()->getNumNeurons() * s.get().getTrgNeuronGroup()->getNumDelaySlots();
            const auto wuPostVars = wu->getPostVars();
            for(size_t i = 0; i < wuPostVars.size(); i++) {
                const bool autoInitialized = !s.get().getWUPostVarInitialisers()[i].getSnippet()->getCode().empty();
                mem += genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                runnerPushFunc, runnerPullFunc, wuPostVars[i].type, wuPostVars[i].name + s.get().getName(),
                                s.get().getWUPostVarLocation(i), autoInitialized, postSize, synapseGroupStatePushPullFunctions);
            }

            // If this synapse group's postsynaptic models hasn't been merged (which makes pulling them somewhat ambiguous)
            // **NOTE** we generated initialisation and declaration code earlier - here we just generate push and pull as we want this per-synapse group
            if(!s.get().isPSModelMerged()) {
                // Add code to push and pull inSyn
                genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.get().getInSynLocation(), "inSyn" + s.get().getName(), synapseGroupStatePushPullFunctions,
                    [&]()
                    {
                        backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, model.getPrecision(), "inSyn" + s.get().getName(), s.get().getInSynLocation(),
                                                    true, s.get().getTrgNeuronGroup()->getNumNeurons());
                    });

                // If this synapse group has individual postsynaptic model variables
                if (s.get().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    const auto psmVars = psm->getVars();
                    for(size_t i = 0; i < psmVars.size(); i++) {
                        const bool autoInitialized = !s.get().getPSVarInitialisers()[i].getSnippet()->getCode().empty();
                        genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.get().getPSVarLocation(i), psmVars[i].name + s.get().getName(), synapseGroupStatePushPullFunctions,
                            [&]()
                            {
                                backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, psmVars[i].type, psmVars[i].name + s.get().getName(), s.get().getPSVarLocation(i),
                                                            autoInitialized, s.get().getTrgNeuronGroup()->getNumNeurons());
                            });
                    }
                }
            }

            // Add helper function to push and pull entire synapse group state
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, s.get().getName(), synapseGroupStatePushPullFunctions);

            const auto psmExtraGlobalParams = psm->getExtraGlobalParams();
            for(size_t i = 0; i < psmExtraGlobalParams.size(); i++) {
                genExtraGlobalParam(backend, definitionsVar, definitionsFunc, runnerVarDecl, runnerExtraGlobalParamFunc,
                                    psmExtraGlobalParams[i].type, psmExtraGlobalParams[i].name + s.get().getName(), s.get().getPSExtraGlobalParamLocation(i));
            }

            const auto wuExtraGlobalParams = wu->getExtraGlobalParams();
            for(size_t i = 0; i < wuExtraGlobalParams.size(); i++) {
                genExtraGlobalParam(backend, definitionsVar, definitionsFunc, runnerVarDecl, runnerExtraGlobalParamFunc,
                                    wuExtraGlobalParams[i].type, wuExtraGlobalParams[i].name + s.get().getName(), s.get().getWUExtraGlobalParamLocation(i));
            }

            const auto sparseConnExtraGlobalParams = s.get().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams();
            for(size_t i = 0; i < sparseConnExtraGlobalParams.size(); i++) {
                genExtraGlobalParam(backend, definitionsVar, definitionsFunc, runnerVarDecl, runnerExtraGlobalParamFunc,
                                    sparseConnExtraGlobalParams[i].type, sparseConnExtraGlobalParams[i].name + s.get().getName(),
                                    s.get().getSparseConnectivityExtraGlobalParamLocation(i));
            }
        }
    }
    allVarStreams << std::endl;

    definitionsInternal << "// ------------------------------------------------------------------------" << std::endl;
    definitionsInternal << "// merged group structures" << std::endl;
    definitionsInternal << "// ------------------------------------------------------------------------" << std::endl;

    definitionsInternalFunc << "// ------------------------------------------------------------------------" << std::endl;
    definitionsInternalFunc << "// copying merged group structures to device" << std::endl;
    definitionsInternalFunc << "// ------------------------------------------------------------------------" << std::endl;

    const std::string hack = "d_";

    // Loop through merged neuron initialisation groups
    for(const auto &m : model.getMergedLocalNeuronInitGroups()) {
        MergedStructGenerator<NeuronGroupMerged> gen;

        const NeuronModels::Base *nm = m.getArchetype().getNeuronModel();

        const bool populationRNG = m.getArchetype().isSimRNGRequired();
        const bool trueSpike = !m.getArchetype().getNeuronModel()->getThresholdConditionCode().empty();
        const bool spikeLikeEvent = m.getArchetype().isSpikeEventRequired();
        const bool delay = m.getArchetype().isDelayRequired();

        // Write struct declation to top of definitions internal
        definitionsInternal << "struct MergedNeuronInitGroup" << m.getIndex() << std::endl;
        {
            CodeStream::Scope b(definitionsInternal);
            definitionsInternal << "unsigned int numNeurons;" << std::endl;
            definitionsInternal << std::endl;

            // Add spike arrays
            if(trueSpike) {
                definitionsInternal << "// Spikes" << std::endl;
                definitionsInternal << "unsigned int* spkCnt;" << std::endl;
                definitionsInternal << "unsigned int* spk;" << std::endl;
                definitionsInternal << std::endl;
            }

            // Add spike like event arrays
            if(spikeLikeEvent) {
                definitionsInternal << "// Spike-like events" << std::endl;
                definitionsInternal << "unsigned int* spkCntEvnt;" << std::endl;
                definitionsInternal << "unsigned int* spkEvnt;" << std::endl;
                definitionsInternal << std::endl;
            }

            // Add RNG state reference
            if(populationRNG) {
                definitionsInternal << "// Population RNG" << std::endl;
                definitionsInternal << "curandState* rng;" << std::endl;
                definitionsInternal << std::endl;
            }

            // Add pointers to var pointers to struct
            definitionsInternal << "// Variables" << std::endl;
            for(const auto &v : nm->getVars()) {
                definitionsInternal << v.type << "* " << v.name << ";" << std::endl;
            }
            definitionsInternal << std::endl;

            // Loop through merged synaptic inputs in archetypical neuron group
            for(size_t i = 0; i < m.getArchetype().getMergedInSyn().size(); i++) {
                const SynapseGroupInternal *sg = m.getArchetype().getMergedInSyn()[i].first;
                definitionsInternal << "// Synaptic input " << i << std::endl;

                // Add pointer to insyn
                definitionsInternal << model.getPrecision() << "* inSyn" << i << ";" << std::endl;

                // Add pointer to dendritic delay buffer if required
                if (sg->isDendriticDelayRequired()) {
                    definitionsInternal << model.getPrecision() << "* denDelay" << i << ";" << std::endl;
                }

                // Add pointers to state variables
                if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                    for (const auto &v : sg->getPSModel()->getVars()) {
                        definitionsInternal << v.type << "* " << v.name << i << ";" << std::endl;
                    }
                }
                definitionsInternal << std::endl;
            }
        }

        definitionsInternal << ";" << std::endl;

        // Write local array of these structs containing individual neuron group pointers etc
        runnerVarAlloc << "MergedNeuronInitGroup" << m.getIndex() << " mergedNeuronInitGroup" << m.getIndex() << "[] = ";
        {
            CodeStream::Scope b(runnerVarAlloc);
            for(const auto &ng : m.getGroups()) {
                runnerVarAlloc << "{";
                runnerVarAlloc << ng.get().getNumNeurons() << ", ";

                if(trueSpike) {
                    runnerVarAlloc << hack << "glbSpkCnt" << ng.get().getName() << ", ";
                    runnerVarAlloc << hack << "glbSpk" << ng.get().getName() << ", ";
                }

                if(spikeLikeEvent) {
                    runnerVarAlloc << hack << "glbSpkCntEvnt" << ng.get().getName() << ", ";
                    runnerVarAlloc << hack << "glbSpkEvnt" << ng.get().getName() << ", ";
                }

                if(delay) {
                    runnerVarAlloc << hack << "spkQuePtr" << ng.get().getName() << ", ";
                }

                if(populationRNG) {
                    runnerVarAlloc << hack << "rng" << ng.get().getName() << ", ";
                }

                for(const auto &v : nm->getVars()) {
                    runnerVarAlloc << hack << v.name << ng.get().getName() << ", ";
                }

                // Loop through merged synaptic inputs in the archetypical neuron group
                for (const auto &m : m.getArchetype().getMergedInSyn()) {
                    const SynapseGroupInternal *archetypeSG = m.first;

                    // Find merged synapse group in THIS neuron group which is compatible with archetype
                    const auto otherSyn = std::find_if(ng.get().getMergedInSyn().cbegin(), ng.get().getMergedInSyn().cend(),
                                                       [archetypeSG](const std::pair<SynapseGroupInternal*, std::vector<SynapseGroupInternal*>> &m)
                                                       {
                                                           return m.first->canPSBeMerged(*archetypeSG);
                                                       });
                    assert(otherSyn != ng.get().getMergedInSyn().cend());

                    runnerVarAlloc << hack << "inSyn" << otherSyn->first->getPSModelTargetName() << ", ";

                    // Add pointer to dendritic delay buffer if required
                    if (otherSyn->first->isDendriticDelayRequired()) {
                        runnerVarAlloc << hack << "denDelay" << otherSyn->first->getPSModelTargetName() << ", ";
                    }

                    // Add pointers to state variables
                    if (otherSyn->first->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                        for (const auto &v : otherSyn->first->getPSModel()->getVars()) {
                            runnerVarAlloc << hack << v.name << otherSyn->first->getPSModelTargetName() << ", ";
                        }
                    }

                }

                runnerVarAlloc << "}," << std::endl;

            }
        }
        runnerVarAlloc << ";" << std::endl;

        // Then generate call to function to copy local array to device
        runnerVarAlloc << "pushMergedNeuronInitGroup" << m.getIndex() << "ToDevice(mergedNeuronInitGroup" << m.getIndex() << ");" << std::endl;

        // Finally add declaration to function to definitions internal
        definitionsInternalFunc << "EXPORT_FUNC void pushMergedNeuronInitGroup" << m.getIndex() << "ToDevice(const MergedNeuronInitGroup" << m.getIndex() << " *group);" << std::endl;

    }

    // Loop through merged synapse connectivity initialisation groups
    for(const auto &m : model.getMergedLocalSynapseConnectivityInitGroups()) {
        MergedStructGenerator<SynapseGroupMerged> gen;

        gen.addField("unsigned int numSrcNeurons",
                     [](const SynapseGroupInternal &sg){ return std::to_string(sg.getSrcNeuronGroup()->getNumNeurons()); });
        gen.addField("unsigned int numTrgNeurons",
                     [](const SynapseGroupInternal &sg){ return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });
        gen.addField("unsigned int rowStride",
                     [&backend, m](const SynapseGroupInternal &sg){ return std::to_string(backend.getSynapticMatrixRowStride(m, sg)); });

        if(m.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            gen.addField("unsigned int *rowLength",
                         [hack](const SynapseGroupInternal &sg){ return hack + "rowLength" + sg.getName(); });
            gen.addField(m.getArchetype().getSparseIndType() + " *ind",
                         [hack](const SynapseGroupInternal &sg){ return hack + "ind" + sg.getName(); });
        }
        else if(m.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            gen.addField("uint32_t *gp",
                         [hack](const SynapseGroupInternal &sg){ return hack + "gp" + sg.getName(); });
        }

        // Add EGPs to struct
        gen.addEGPs(m.getArchetype().getConnectivityInitialiser().getSnippet()->getExtraGlobalParams());

        // Generate structure definitions and instantiation
        gen.generate(definitionsInternal, definitionsInternalFunc, runnerVarAlloc, "SynapseConnectivityInit", m);
    }

    // Loop through merged neuron groups
    for(const auto &m : model.getMergedLocalNeuronGroups()) {
        MergedStructGenerator<NeuronGroupMerged> gen;
        gen.addField("unsigned int numNeurons",
                     [](const NeuronGroupInternal &ng){ return std::to_string(ng.getNumNeurons()); });

        gen.addField("unsigned int *spkCnt",
                     [hack](const NeuronGroupInternal &ng){ return hack + "glbSpkCnt" + ng.getName(); });
        gen.addField("unsigned int *spk",
                     [hack](const NeuronGroupInternal &ng){ return hack + "glbSpk" + ng.getName(); });

        if(m.getArchetype().isSpikeEventRequired()) {
            gen.addField("unsigned int *spkCntEvnt",
                         [hack](const NeuronGroupInternal &ng){ return hack + "glbSpkCntEvnt" + ng.getName(); });
            gen.addField("unsigned int *spkEvnt",
                         [hack](const NeuronGroupInternal &ng){ return hack + "glbSpkEvnt" + ng.getName(); });
        }

        if(m.getArchetype().isDelayRequired()) {
            gen.addField("volatile unsigned int *spkQuePtr",
                         [hack](const NeuronGroupInternal &ng){ return "&" + hack + "spkQuePtr" + ng.getName(); });
        }

        if(m.getArchetype().isSimRNGRequired()) {
            gen.addField("curandState *rng",
                         [hack](const NeuronGroupInternal &ng){ return hack + "rng" + ng.getName(); });
        }

        // Add pointers to variables
        const NeuronModels::Base *nm = m.getArchetype().getNeuronModel();
        gen.addVars(nm->getVars(), hack);
        gen.addEGPs(nm->getExtraGlobalParams());

        // Loop through merged synaptic inputs in archetypical neuron group
        for(size_t i = 0; i < m.getArchetype().getMergedInSyn().size(); i++) {
            const SynapseGroupInternal *sg = m.getArchetype().getMergedInSyn()[i].first;

            // Add pointer to insyn
            gen.addField(model.getPrecision() + " *inSyn" + std::to_string(i),
                         [i, m, hack](const NeuronGroupInternal &ng)
                         {
                             return hack + "inSyn" + m.getCompatibleMergedInSyn(i, ng)->getPSModelTargetName();
                         });

            // Add pointer to dendritic delay buffer if required
            if (sg->isDendriticDelayRequired()) {
                gen.addField(model.getPrecision() + " *denDelay" + std::to_string(i),
                             [i, m, hack](const NeuronGroupInternal &ng)
                             {
                                return hack + "denDelay" + m.getCompatibleMergedInSyn(i, ng)->getPSModelTargetName();
                             });

                gen.addField("volatile unsigned int *denDelayPtr" + std::to_string(i),
                             [i, m, hack, &backend](const NeuronGroupInternal &ng)
                             {
                                return "getSymbolAddress(" + backend.getVarPrefix() + "denDelayPtr" + m.getCompatibleMergedInSyn(i, ng)->getPSModelTargetName() + ")";
                             });
            }

            // Add pointers to state variables
            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                for (const auto &v : sg->getPSModel()->getVars()) {
                    gen.addField(v.type + " *" + v.name + std::to_string(i),
                             [i, m, hack, v](const NeuronGroupInternal &ng)
                             {
                                return hack + v.name + m.getCompatibleMergedInSyn(i, ng)->getPSModelTargetName();
                             });
                }
            }
        }

        // Generate structure definitions and instantiation
        gen.generate(definitionsInternal, definitionsInternalFunc, runnerVarAlloc, "Neuron", m);
    }

    // Loop through merged synapse groups
    for(const auto &m : model.getMergedLocalSynapseGroups()) {
        MergedStructGenerator<SynapseGroupMerged> gen;

        gen.addField("unsigned int rowStride",
                     [m, &backend](const SynapseGroupInternal &sg){ return std::to_string(backend.getSynapticMatrixRowStride(m, sg)); });
        gen.addField("unsigned int numTrgNeurons",
                     [](const SynapseGroupInternal &sg){ return std::to_string(sg.getTrgNeuronGroup()->getNumNeurons()); });

        if(m.getArchetype().isDendriticDelayRequired()) {
            gen.addField(model.getPrecision() + " *denDelay",
                         [hack](const SynapseGroupInternal &sg){ return hack + "denDelay" + sg.getPSModelTargetName(); });
            gen.addField("volatile unsigned int *denDelayPtr",
                         [&backend](const SynapseGroupInternal &sg){ return "getSymbolAddress(" + backend.getVarPrefix() + "denDelayPtr" + sg.getPSModelTargetName() + ")"; });
        }
        else {
            gen.addField(model.getPrecision() + " *inSyn",
                         [hack](const SynapseGroupInternal &sg){ return hack + "inSyn" + sg.getPSModelTargetName(); });
        }

        if(m.getArchetype().isTrueSpikeRequired()) {
            gen.addField("unsigned int* preSpkCnt",
                         [hack](const SynapseGroupInternal &sg){ return hack + "glbSpkCnt" + sg.getSrcNeuronGroup()->getName(); });
            gen.addField("unsigned int* preSpk",
                         [hack](const SynapseGroupInternal &sg){ return hack + "glbSpk" + sg.getSrcNeuronGroup()->getName(); });
        }

        if(m.getArchetype().isSpikeEventRequired()) {
            gen.addField("unsigned int* preSpkCntEvnt",
                         [hack](const SynapseGroupInternal &sg){ return hack + "glbSpkCntEvnt" + sg.getSrcNeuronGroup()->getName(); });
            gen.addField("unsigned int* preSpkEvnt",
                         [hack](const SynapseGroupInternal &sg){ return hack + "glbSpkEvnt" + sg.getSrcNeuronGroup()->getName(); });
        }

         // Add pointers to connectivity data
        if(m.getArchetype().getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
            gen.addField("unsigned int *rowLength",
                         [hack](const SynapseGroupInternal &sg){ return hack + "rowLength" + sg.getName(); });
            gen.addField(m.getArchetype().getSparseIndType() + " *ind",
                         [hack](const SynapseGroupInternal &sg){ return hack + "ind" + sg.getName(); });
        }
        else if(m.getArchetype().getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            gen.addField("uint32_t *gp",
                         [hack](const SynapseGroupInternal &sg){ return hack + "gp" + sg.getName(); });
        }

        // Add pointers to var pointers to struct
        const WeightUpdateModels::Base *wum = m.getArchetype().getWUModel();
        if(m.getArchetype().getMatrixType() & SynapseMatrixWeight::INDIVIDUAL) {
            gen.addVars(wum->getVars(), hack);
        }

        // Add EGPs to struct
        gen.addEGPs(wum->getExtraGlobalParams());

        // Generate structure definitions and instantiation
        gen.generate(definitionsInternal, definitionsInternalFunc, runnerVarAlloc, "Synapse", m);
    }

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

    runner << "// ------------------------------------------------------------------------" << std::endl;
    runner << "// helper getter functions" << std::endl;
    runner << "// ------------------------------------------------------------------------" << std::endl;
    runner << runnerGetterFuncStream.str();
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
    definitionsInternal << definitionsInternalVarStream.str();
    definitionsInternal << definitionsInternalFuncStream.str();

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
