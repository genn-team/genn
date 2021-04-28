#include "code_generator/generateRunner.h"

// Standard C++ includes
#include <random>
#include <set>
#include <sstream>
#include <string>

// GeNN includes
#include "gennUtils.h"

// GeNN code generator
#include "code_generator/codeGenUtils.h"
#include "code_generator/codeStream.h"
#include "code_generator/groupMerged.h"
#include "code_generator/substitutions.h"
#include "code_generator/teeStream.h"
#include "code_generator/backendBase.h"
#include "code_generator/modelSpecMerged.h"

using namespace CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
unsigned int getNumCopies(VarAccess varAccess, unsigned int batchSize)
{
    return (varAccess & VarAccessDuplication::SHARED) ? 1 : batchSize;
}
//--------------------------------------------------------------------------
void genSpikeMacros(CodeStream &os, const NeuronGroupInternal &ng, bool trueSpike)
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
void genHostScalar(CodeStream &definitionsVar, CodeStream &runnerVarDecl, const std::string &type, const std::string &name, const std::string &value)
{
    definitionsVar << "EXPORT_VAR " << type << " " << name << ";" << std::endl;
    runnerVarDecl << type << " " << name << " = " << value << ";" << std::endl;
}
//--------------------------------------------------------------------------
void genHostDeviceScalar(const BackendBase &backend, CodeStream &definitionsVar, CodeStream &definitionsInternalVar,
                         CodeStream &runnerVarDecl, CodeStream &runnerVarAlloc, CodeStream &runnerVarFree,
                         const std::string &type, const std::string &name, const std::string &hostValue, MemAlloc &mem)
{
    // Generate a host scalar
    genHostScalar(definitionsVar, runnerVarDecl, type, name, hostValue);

    // Generate a single-element array on device
    if(backend.isDeviceScalarRequired()) {
        backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                type, name, VarLocation::DEVICE, 1, mem);
    }
}
//--------------------------------------------------------------------------
bool canPushPullVar(VarLocation loc)
{
    // A variable can be pushed and pulled if it is located on both host and device
    return ((loc & VarLocation::HOST) &&
            (loc & VarLocation::DEVICE));
}
//-------------------------------------------------------------------------
bool genVarPushPullScope(CodeStream &definitionsFunc, CodeStream &runnerPushFunc, CodeStream &runnerPullFunc,
                         VarLocation loc, bool automaticCopyEnabled, const std::string &description, std::function<void()> handler)
{
    // If this variable has a location that allows pushing and pulling and automatic copying isn't enabled
    if(canPushPullVar(loc) && !automaticCopyEnabled) {
        definitionsFunc << "EXPORT_FUNC void push" << description << "ToDevice(bool uninitialisedOnly = false);" << std::endl;
        definitionsFunc << "EXPORT_FUNC void pull" << description << "FromDevice();" << std::endl;

        runnerPushFunc << "void push" << description << "ToDevice(bool uninitialisedOnly)";
        runnerPullFunc << "void pull" << description << "FromDevice()";
        {
            CodeStream::Scope a(runnerPushFunc);
            CodeStream::Scope b(runnerPullFunc);

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
void genVarPushPullScope(CodeStream &definitionsFunc, CodeStream &runnerPushFunc, CodeStream &runnerPullFunc,
                         VarLocation loc, bool automaticCopyEnabled, const std::string &description, std::vector<std::string> &statePushPullFunction,
                         std::function<void()> handler)
{
    // Add function to vector if push pull function was actually required
    if(genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, loc, automaticCopyEnabled, description, handler)) {
        statePushPullFunction.push_back(description);
    }
}
//-------------------------------------------------------------------------
void genVarGetterScope(CodeStream &definitionsFunc, CodeStream &runnerGetterFunc,
                       VarLocation loc, const std::string &description, 
                       const std::string &type, std::function<void()> handler)
{
    // If this variable has a location that allows pushing and pulling and hence getting a host pointer
    if(canPushPullVar(loc)) {
        // Export getter
        definitionsFunc << "EXPORT_FUNC " << type << " get" << description << "(unsigned int batch = 0); " << std::endl;

        // Define getter
        runnerGetterFunc << type << " get" << description << "(" << "unsigned int batch" << ")";
        {
            CodeStream::Scope a(runnerGetterFunc);
            handler();
        }
        runnerGetterFunc << std::endl;
    }
}
//-------------------------------------------------------------------------
void genSpikeGetters(CodeStream &definitionsFunc, CodeStream &runnerGetterFunc,
                     const NeuronGroupInternal &ng, bool trueSpike, unsigned int batchSize)
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
                          runnerGetterFunc << "return (glbSpk" << eventSuffix << ng.getName();
                          if (delayRequired) {
                              runnerGetterFunc << " + (spkQuePtr" << ng.getName() << " * " << ng.getNumNeurons() << ")";
                              if(batchSize > 1) {
                                  runnerGetterFunc << " + (batch * " << ng.getNumNeurons() * ng.getNumDelaySlots() << ")";
                              }
                          }
                          else if(batchSize > 1) {
                              runnerGetterFunc << " + (batch * " << ng.getNumNeurons() << ")";
                          }
                          runnerGetterFunc << ");" << std::endl;
                      });

    // Generate getter for current spikes
    genVarGetterScope(definitionsFunc, runnerGetterFunc,
                      loc, ng.getName() + (trueSpike ? "CurrentSpikeCount" : "CurrentSpikeEventCount"), "unsigned int&",
                      [&]()
                      {
                          runnerGetterFunc << "return glbSpkCnt" << eventSuffix << ng.getName() << "[";
                          if (delayRequired) {
                              runnerGetterFunc << "spkQuePtr" << ng.getName();
                              if(batchSize > 1) {
                                  runnerGetterFunc << " + (batch * " << ng.getNumDelaySlots() << ")";
                              }
                          }
                          else {
                              if(batchSize == 1) {
                                  runnerGetterFunc << "0";
                              }
                              else {
                                  runnerGetterFunc << "batch";
                              }
                          }
                          runnerGetterFunc << "];" << std::endl;
                      });


}
//-------------------------------------------------------------------------
void genStatePushPull(CodeStream &definitionsFunc, CodeStream &runnerPushFunc, CodeStream &runnerPullFunc,
                      const std::string &name, bool generateEmptyStatePushPull, 
                      const std::vector<std::string> &groupPushPullFunction, std::vector<std::string> &modelPushPullFunctions)
{
    // If we should either generate emtpy state push pull functions or this one won't be empty!
    if(generateEmptyStatePushPull || !groupPushPullFunction.empty()) {
        definitionsFunc << "EXPORT_FUNC void push" << name << "StateToDevice(bool uninitialisedOnly = false);" << std::endl;
        definitionsFunc << "EXPORT_FUNC void pull" << name << "StateFromDevice();" << std::endl;

        runnerPushFunc << "void push" << name << "StateToDevice(bool uninitialisedOnly)";
        runnerPullFunc << "void pull" << name << "StateFromDevice()";
        {
            CodeStream::Scope a(runnerPushFunc);
            CodeStream::Scope b(runnerPullFunc);

            for(const auto &func : groupPushPullFunction) {
                runnerPushFunc << "push" << func << "ToDevice(uninitialisedOnly);" << std::endl;
                runnerPullFunc << "pull" << func << "FromDevice();" << std::endl;
            }
        }
        runnerPushFunc << std::endl;
        runnerPullFunc << std::endl;

        // Add function to list
        modelPushPullFunctions.push_back(name);
    }
}
//-------------------------------------------------------------------------
void genVariable(const BackendBase &backend, CodeStream &definitionsVar, CodeStream &definitionsFunc,
                     CodeStream &definitionsInternal, CodeStream &runner, CodeStream &allocations, CodeStream &free,
                     CodeStream &push, CodeStream &pull, const std::string &type, const std::string &name,
                     VarLocation loc, bool autoInitialized, size_t count, MemAlloc &mem,
                     std::vector<std::string> &statePushPullFunction)
{
    // Generate push and pull functions
    genVarPushPullScope(definitionsFunc, push, pull, loc, backend.getPreferences().automaticCopy, name, statePushPullFunction,
        [&]()
        {
            backend.genVariablePushPull(push, pull, type, name, loc, autoInitialized, count);
        });

    // Generate variables
    backend.genArray(definitionsVar, definitionsInternal, runner, allocations, free,
                     type, name, loc, count, mem);
}
//-------------------------------------------------------------------------
void genExtraGlobalParam(const ModelSpecMerged &modelMerged, const BackendBase &backend, CodeStream &definitionsVar,
                         CodeStream &definitionsFunc, CodeStream &definitionsInternalVar, CodeStream &runner,
                         CodeStream &extraGlobalParam, const std::string &type, const std::string &name, bool apiRequired, VarLocation loc)
{
    // Generate variables
    backend.genExtraGlobalParamDefinition(definitionsVar, definitionsInternalVar, type, name, loc);
    backend.genExtraGlobalParamImplementation(runner, type, name, loc);

    // If type is a pointer and API is required
    if(Utils::isTypePointer(type) && apiRequired) {
        // Write definitions for functions to allocate and free extra global param
        definitionsFunc << "EXPORT_FUNC void allocate" << name << "(unsigned int count);" << std::endl;
        definitionsFunc << "EXPORT_FUNC void free" << name << "();" << std::endl;

        // Write allocation function
        extraGlobalParam << "void allocate" << name << "(unsigned int count)";
        {
            CodeStream::Scope a(extraGlobalParam);
            backend.genExtraGlobalParamAllocation(extraGlobalParam, type, name, loc);

            // Get destinations in merged structures, this EGP 
            // needs to be copied to and call push function
            const auto &mergedDestinations = modelMerged.getMergedEGPDestinations(name, backend);
            for(const auto &v : mergedDestinations) {
                extraGlobalParam << "pushMerged" << v.first << v.second.mergedGroupIndex << v.second.fieldName << "ToDevice(";
                extraGlobalParam << v.second.groupIndex << ", " << backend.getDeviceVarPrefix() << name << ");" << std::endl;
            }
        }

        // Write free function
        extraGlobalParam << "void free" << name << "()";
        {
            CodeStream::Scope a(extraGlobalParam);
            backend.genVariableFree(extraGlobalParam, name, loc);
        }

        // If variable can be pushed and pulled
        if(!backend.getPreferences().automaticCopy && canPushPullVar(loc)) {
            // Write definitions for push and pull functions
            definitionsFunc << "EXPORT_FUNC void push" << name << "ToDevice(unsigned int count);" << std::endl;

            // Write push function
            extraGlobalParam << "void push" << name << "ToDevice(unsigned int count)";
            {
                CodeStream::Scope a(extraGlobalParam);
                backend.genExtraGlobalParamPush(extraGlobalParam, type, name, loc);
            }

            if(backend.getPreferences().generateExtraGlobalParamPull) {
                // Write definitions for pull functions
                definitionsFunc << "EXPORT_FUNC void pull" << name << "FromDevice(unsigned int count);" << std::endl;

                // Write pull function
                extraGlobalParam << "void pull" << name << "FromDevice(unsigned int count)";
                {
                    CodeGenerator::CodeStream::Scope a(extraGlobalParam);
                    backend.genExtraGlobalParamPull(extraGlobalParam, type, name, loc);
                }
            }
        }

    }
}
//-------------------------------------------------------------------------
void genGlobalHostRNG(CodeStream &definitionsVar, CodeStream &runnerVarDecl,
                      CodeStream &runnerVarAlloc, unsigned int seed, MemAlloc &mem)
{
    definitionsVar << "EXPORT_VAR " << "std::mt19937 hostRNG;" << std::endl;
    runnerVarDecl << "std::mt19937 hostRNG;" << std::endl;

    // If no seed is specified, use system randomness to generate seed sequence
    CodeStream::Scope b(runnerVarAlloc);
    if(seed == 0) {
        runnerVarAlloc << "uint32_t seedData[std::mt19937::state_size];" << std::endl;
        runnerVarAlloc << "std::random_device seedSource;" << std::endl;
        runnerVarAlloc << "for(int i = 0; i < std::mt19937::state_size; i++)";
        {
            CodeStream::Scope b(runnerVarAlloc);
            runnerVarAlloc << "seedData[i] = seedSource();" << std::endl;
        }
        runnerVarAlloc << "std::seed_seq seeds(std::begin(seedData), std::end(seedData));" << std::endl;
    }
    // Otherwise, create a seed sequence from model seed
    // **NOTE** this is a terrible idea see http://www.pcg-random.org/posts/cpp-seeding-surprises.html
    else {
        runnerVarAlloc << "std::seed_seq seeds{" << seed << "};" << std::endl;
    }

    // Seed RNG from seed sequence
    runnerVarAlloc << "hostRNG.seed(seeds);" << std::endl;

    // Add size of Mersenne Twister to memory tracker
    mem += MemAlloc::host(sizeof(std::mt19937));
}
//-------------------------------------------------------------------------
void genSynapseConnectivityHostInit(const BackendBase &backend, CodeStream &os, 
                                    const SynapseConnectivityHostInitGroupMerged &sg, const std::string &precision)
{
    CodeStream::Scope b(os);
    os << "// merged synapse connectivity host init group " << sg.getIndex() << std::endl;
    os << "for(unsigned int g = 0; g < " << sg.getGroups().size() << "; g++)";
    {
        CodeStream::Scope b(os);

        // Get reference to group
        os << "const auto *group = &mergedSynapseConnectivityHostInitGroup" << sg.getIndex() << "[g]; " << std::endl;

        const auto &connectInit = sg.getArchetype().getConnectivityInitialiser();

        // If matrix type is procedural then initialized connectivity init snippet will potentially be used with multiple threads per spike. 
        // Otherwise it will only ever be used for initialization which uses one thread per row
        const size_t numThreads = (sg.getArchetype().getMatrixType() & SynapseMatrixConnectivity::PROCEDURAL) ? sg.getArchetype().getNumThreadsPerSpike() : 1;

        // Create substitutions
        Substitutions subs;
        subs.addVarSubstitution("rng", "hostRNG");
        subs.addVarSubstitution("num_pre", "group->numSrcNeurons");
        subs.addVarSubstitution("num_post", "group->numTrgNeurons");
        subs.addVarSubstitution("num_threads", std::to_string(numThreads));
        subs.addVarNameSubstitution(connectInit.getSnippet()->getExtraGlobalParams(), "", "*group->");
        subs.addParamValueSubstitution(connectInit.getSnippet()->getParamNames(), connectInit.getParams(),
                                       [&sg](size_t p) { return sg.isConnectivityInitParamHeterogeneous(p); },
                                       "", "group->");
        subs.addVarValueSubstitution(connectInit.getSnippet()->getDerivedParams(), connectInit.getDerivedParams(),
                                     [&sg](size_t p) { return sg.isConnectivityInitDerivedParamHeterogeneous(p); },
                                     "", "group->");

        // Loop through EGPs
        const auto egps = connectInit.getSnippet()->getExtraGlobalParams();
        for(size_t i = 0; i < egps.size(); i++) {
            const auto loc = sg.getArchetype().getSparseConnectivityExtraGlobalParamLocation(i);
            // If EGP is a pointer and located on the host
            if(Utils::isTypePointer(egps[i].type) && (loc & VarLocation::HOST)) {
                // Generate code to allocate this EGP with count specified by $(0)
                std::stringstream allocStream;
                CodeGenerator::CodeStream alloc(allocStream);
                backend.genExtraGlobalParamAllocation(alloc, egps[i].type + "*", egps[i].name,
                                                      loc, "$(0)", "group->");

                // Add substitution
                subs.addFuncSubstitution("allocate" + egps[i].name, 1, allocStream.str());

                // Generate code to push this EGP with count specified by $(0)
                std::stringstream pushStream;
                CodeStream push(pushStream);
                backend.genExtraGlobalParamPush(push, egps[i].type + "*", egps[i].name,
                                                loc, "$(0)", "group->");


                // Add substitution
                subs.addFuncSubstitution("push" + egps[i].name, 1, pushStream.str());
            }
        }
        std::string code = connectInit.getSnippet()->getHostInitCode();
        subs.applyCheckUnreplaced(code, "hostInitSparseConnectivity : merged" + std::to_string(sg.getIndex()));
        code = ensureFtype(code, precision);

        // Write out code
        os << code << std::endl;

    }
}
//-------------------------------------------------------------------------
template<typename V, typename S>
void genCustomUpdate(const ModelSpecMerged &modelMerged, const BackendBase &backend, 
                     CodeStream &definitionsVar, CodeStream &definitionsFunc, CodeStream &definitionsInternalVar,
                     CodeStream &runnerVarDecl, CodeStream &runnerVarAlloc, CodeStream &runnerVarFree, CodeStream &runnerExtraGlobalParamFunc,
                     CodeStream &runnerPushFunc, CodeStream &runnerPullFunc, const std::map<std::string, V> &customUpdates,
                     MemAlloc &mem, std::vector<std::string> &statePushPullFunctions, S getSizeFn)
{
    // Loop through customupdates
    for(const auto &c : customUpdates) {
        const auto cuModel = c.second.getCustomUpdateModel();
        const auto cuVars = cuModel->getVars();

        std::vector<std::string> customUpdateStatePushPullFunctions;
        for(size_t i = 0; i < cuVars.size(); i++) {
            const auto *varInitSnippet = c.second.getVarInitialisers()[i].getSnippet();
            const unsigned int numCopies = c.second.isBatched() ? getNumCopies(cuVars[i].access, modelMerged.getModel().getBatchSize()) : 1;
            const bool autoInitialized = !varInitSnippet->getCode().empty();
            genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                        runnerPushFunc, runnerPullFunc, cuVars[i].type, cuVars[i].name + c.first, c.second.getVarLocation(i),
                        autoInitialized, numCopies * getSizeFn(c.second), mem, customUpdateStatePushPullFunctions);

            // Loop through EGPs required to initialize custom update variable
            const auto extraGlobalParams = varInitSnippet->getExtraGlobalParams();
            for(size_t e = 0; e < extraGlobalParams.size(); e++) {
                genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                    runnerVarDecl, runnerExtraGlobalParamFunc,
                                    extraGlobalParams[e].type, extraGlobalParams[e].name + cuVars[i].name + c.first,
                                    true, VarLocation::HOST_DEVICE);
            }
        }

        // Add helper function to push and pull entire custom update state
        if(!backend.getPreferences().automaticCopy) {
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc,
                                c.first, backend.getPreferences().generateEmptyStatePushPull,
                                customUpdateStatePushPullFunctions, statePushPullFunctions);
        }

        const auto csExtraGlobalParams = cuModel->getExtraGlobalParams();
        for(size_t i = 0; i < csExtraGlobalParams.size(); i++) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc,
                                csExtraGlobalParams[i].type, csExtraGlobalParams[i].name + c.first,
                                true, VarLocation::HOST_DEVICE);
        }
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
MemAlloc CodeGenerator::generateRunner(CodeStream &definitions, CodeStream &definitionsInternal, CodeStream &runner,
                                       const ModelSpecMerged &modelMerged, const BackendBase &backend)
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
    backend.genDefinitionsPreamble(definitions, modelMerged);

    // Write definitions internal preamble
    definitionsInternal << "#pragma once" << std::endl;
    definitionsInternal << "#include \"definitions.h\"" << std::endl << std::endl;
    backend.genDefinitionsInternalPreamble(definitionsInternal, modelMerged);
    
    // write DT macro
    const ModelSpecInternal &model = modelMerged.getModel();
    if (model.getTimePrecision() == "float") {
        definitions << "#define DT " << Utils::writePreciseString(model.getDT()) << "f" << std::endl;
    } else {
        definitions << "#define DT " << Utils::writePreciseString(model.getDT()) << std::endl;
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

    // Create codestreams to generate different sections of runner and definitions
    std::stringstream runnerVarDeclStream;
    std::stringstream runnerVarAllocStream;
    std::stringstream runnerMergedStructAllocStream;
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
    CodeStream runnerMergedStructAlloc(runnerMergedStructAllocStream);
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

    if(model.isRecordingInUse()) {
        runnerVarDecl << "unsigned long long numRecordingTimesteps = 0;" << std::endl;
    }
    // If backend requires a global device RNG to simulate (or initialize) this model
    if(backend.isGlobalDeviceRNGRequired(modelMerged)) {
        backend.genGlobalDeviceRNG(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree, mem);
    }
    // If backend required a global host RNG to simulate (or initialize) this model, generate a standard Mersenne Twister
    if(backend.isGlobalHostRNGRequired(modelMerged)) {
        genGlobalHostRNG(definitionsVar, runnerVarDecl, runnerVarAlloc, model.getSeed(), mem);
    }
    allVarStreams << std::endl;

    // Generate preamble for the final stage of time step
    // **NOTE** this is done now as there can be timing logic here
    backend.genStepTimeFinalisePreamble(runnerStepTimeFinalise, modelMerged);

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// timers" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;

    // Build set containing union of all custom update groupsnames
    std::set<std::string> customUpdateGroups;
    std::transform(model.getCustomUpdates().cbegin(), model.getCustomUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomUpdateValueType &v) { return v.second.getUpdateGroupName(); });
    std::transform(model.getCustomWUUpdates().cbegin(), model.getCustomWUUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomUpdateWUValueType &v) { return v.second.getUpdateGroupName(); });

    // Generate variables to store total elapsed time
    // **NOTE** we ALWAYS generate these so usercode doesn't require #ifdefs around timing code
    genHostScalar(definitionsVar, runnerVarDecl, "double", "initTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, "double", "initSparseTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, "double", "neuronUpdateTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, "double", "presynapticUpdateTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, "double", "postsynapticUpdateTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, "double", "synapseDynamicsTime", "0.0");

    // Generate variables to store total elapsed time for each custom update group
    for(const auto &g : customUpdateGroups) {
        genHostScalar(definitionsVar, runnerVarDecl, "double", "customUpdate" + g + "Time", "0.0");
        genHostScalar(definitionsVar, runnerVarDecl, "double", "customUpdate" + g + "TransposeTime", "0.0");
    }
    
    // If timing is actually enabled
    if(model.isTimingEnabled()) {
        // Create neuron timer
        backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         runnerStepTimeFinalise, "neuronUpdate", true);

        // Add presynaptic update timer
        if(!modelMerged.getMergedPresynapticUpdateGroups().empty()) {
            backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "presynapticUpdate", true);
        }

        // Add postsynaptic update timer if required
        if(!modelMerged.getMergedPostsynapticUpdateGroups().empty()) {
            backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "postsynapticUpdate", true);
        }

        // Add synapse dynamics update timer if required
        if(!modelMerged.getMergedSynapseDynamicsGroups().empty()) {
            backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "synapseDynamics", true);
        }

        // Add timers for each custom update group
        for(const auto &g : customUpdateGroups) {
            backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "customUpdate" + g, false);
            backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "customUpdate" + g + "Transpose", false);
        }

        // Create init timer
        backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         runnerStepTimeFinalise, "init", false);

        // Add sparse initialisation timer
        if(!modelMerged.getMergedSynapseSparseInitGroups().empty()) {
            backend.genTimer(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "initSparse", false);
        }

        allVarStreams << std::endl;
    }

    runnerVarDecl << "// ------------------------------------------------------------------------" << std::endl;
    runnerVarDecl << "// merged group arrays" << std::endl;
    runnerVarDecl << "// ------------------------------------------------------------------------" << std::endl;

    definitionsInternal << "// ------------------------------------------------------------------------" << std::endl;
    definitionsInternal << "// merged group structures" << std::endl;
    definitionsInternal << "// ------------------------------------------------------------------------" << std::endl;

    definitionsInternalVar << "// ------------------------------------------------------------------------" << std::endl;
    definitionsInternalVar << "// merged group arrays for host initialisation" << std::endl;
    definitionsInternalVar << "// ------------------------------------------------------------------------" << std::endl;

    definitionsInternalFunc << "// ------------------------------------------------------------------------" << std::endl;
    definitionsInternalFunc << "// copying merged group structures to device" << std::endl;
    definitionsInternalFunc << "// ------------------------------------------------------------------------" << std::endl;

    // Loop through merged synapse connectivity host initialisation groups
    for(const auto &m : modelMerged.getMergedSynapseConnectivityHostInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through merged synapse connectivity host init groups and generate host init code
    // **NOTE** this is done here so valid pointers get copied straight into subsequent structures and merged EGP system isn't required
    for(const auto &sg : modelMerged.getMergedSynapseConnectivityHostInitGroups()) {
        genSynapseConnectivityHostInit(backend, runnerMergedStructAlloc, sg, model.getPrecision());
    }

    // Generate merged neuron initialisation groups
    for(const auto &m : modelMerged.getMergedNeuronInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Generate merged custom update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomUpdateInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Generate merged custom dense WU update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomWUUpdateDenseInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through merged dense synapse init groups
    for(const auto &m : modelMerged.getMergedSynapseDenseInitGroups()) {
         m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                          runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through merged synapse connectivity initialisation groups
    for(const auto &m : modelMerged.getMergedSynapseConnectivityInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through merged sparse synapse init groups
    for(const auto &m : modelMerged.getMergedSynapseSparseInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Generate merged custom sparse WU update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomWUUpdateSparseInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through merged neuron update groups
    for(const auto &m : modelMerged.getMergedNeuronUpdateGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through merged presynaptic update groups
    for(const auto &m : modelMerged.getMergedPresynapticUpdateGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through merged postsynaptic update groups
    for(const auto &m : modelMerged.getMergedPostsynapticUpdateGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through synapse dynamics groups
    for(const auto &m : modelMerged.getMergedSynapseDynamicsGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through neuron groups whose previous spike times need resetting
    for(const auto &m : modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through neuron groups whose spike queues need resetting
    for(const auto &m : modelMerged.getMergedNeuronSpikeQueueUpdateGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through synapse groups whose dendritic delay pointers need updating
    for(const auto &m : modelMerged.getMergedSynapseDendriticDelayUpdateGroups()) {
       m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                        runnerVarDecl, runnerMergedStructAlloc);
    }
    
    // Loop through custom variable update groups
    for(const auto &m : modelMerged.getMergedCustomUpdateGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through custom WU variable update groups
    for(const auto &m : modelMerged.getMergedCustomUpdateWUGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through custom WU transpose variable update groups
    for(const auto &m : modelMerged.getMergedCustomUpdateTransposeWUGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// local neuron groups" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    const unsigned int batchSize = model.getBatchSize();
    std::vector<std::string> currentSpikePullFunctions;
    std::vector<std::string> currentSpikeEventPullFunctions;
    std::vector<std::string> statePushPullFunctions;
    for(const auto &n : model.getNeuronGroups()) {
        // Write convenience macros to access spikes
        if(batchSize == 1) {
            genSpikeMacros(definitionsVar, n.second, true);
        }

        // True spike variables
        const size_t numNeuronDelaySlots = batchSize * (size_t)n.second.getNumNeurons() * (size_t)n.second.getNumDelaySlots();
        const size_t numSpikeCounts = n.second.isTrueSpikeRequired() ? (batchSize * n.second.getNumDelaySlots()) : batchSize;
        const size_t numSpikes = n.second.isTrueSpikeRequired() ? numNeuronDelaySlots : (batchSize * n.second.getNumNeurons());
        backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeLocation(), numSpikeCounts, mem);
        backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         "unsigned int", "glbSpk" + n.first, n.second.getSpikeLocation(), numSpikes, mem);

        // True spike push and pull functions
        genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeLocation(),
                            backend.getPreferences().automaticCopy, n.first + "Spikes",
                            [&]()
                            {
                                backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                                            "unsigned int", "glbSpkCnt" + n.first, n.second.getSpikeLocation(), true, numSpikeCounts);
                                backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,
                                                            "unsigned int", "glbSpk" + n.first, n.second.getSpikeLocation(), true, numSpikes);
                            });

        // Current true spike push and pull functions
        genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeLocation(),
                            backend.getPreferences().automaticCopy, n.first + "CurrentSpikes", currentSpikePullFunctions,
                            [&]()
                            {
                                backend.genCurrentTrueSpikePush(runnerPushFunc, n.second, batchSize);
                                backend.genCurrentTrueSpikePull(runnerPullFunc, n.second, batchSize);
                            });

        // Current true spike getter functions
        genSpikeGetters(definitionsFunc, runnerGetterFunc, n.second, true, batchSize);

        // If spike recording is enabled, define and declare variables and add free
        if(n.second.isSpikeRecordingEnabled()) {
            backend.genVariableDefinition(definitionsVar, definitionsInternalVar, "uint32_t*", "recordSpk" + n.first, VarLocation::HOST_DEVICE);
            backend.genVariableImplementation(runnerVarDecl, "uint32_t*", "recordSpk" + n.first, VarLocation::HOST_DEVICE);
            backend.genVariableFree(runnerVarFree, "recordSpk" + n.first, VarLocation::HOST_DEVICE);
        }

        // If neuron group needs to emit spike-like events
        if (n.second.isSpikeEventRequired()) {
            // Write convenience macros to access spike-like events
            if(batchSize == 1) {
                genSpikeMacros(definitionsVar, n.second, false);
            }

            // Spike-like event variables
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             "unsigned int", "glbSpkCntEvnt" + n.first, n.second.getSpikeEventLocation(),
                             batchSize * n.second.getNumDelaySlots(), mem);
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             "unsigned int", "glbSpkEvnt" + n.first, n.second.getSpikeEventLocation(),
                              numNeuronDelaySlots, mem);

            // Spike-like event push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeEventLocation(),
                                backend.getPreferences().automaticCopy, n.first + "SpikeEvents",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, "unsigned int", "glbSpkCntEvnt" + n.first, 
                                                                n.second.getSpikeLocation(), true, batchSize * n.second.getNumDelaySlots());
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, "unsigned int", "glbSpkEvnt" + n.first, 
                                                                n.second.getSpikeLocation(), true, numNeuronDelaySlots);
                                });

            // Current spike-like event push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeEventLocation(),
                                backend.getPreferences().automaticCopy, n.first + "CurrentSpikeEvents", currentSpikeEventPullFunctions,
                                [&]()
                                {
                                    backend.genCurrentSpikeLikeEventPush(runnerPushFunc, n.second, batchSize);
                                    backend.genCurrentSpikeLikeEventPull(runnerPullFunc, n.second, batchSize);
                                });

            // Current true spike getter functions
            genSpikeGetters(definitionsFunc, runnerGetterFunc, n.second, false, batchSize);

            // If spike recording is enabled, define and declare variables and add free
            if(n.second.isSpikeEventRecordingEnabled()) {
                backend.genVariableDefinition(definitionsVar, definitionsInternalVar, "uint32_t*", "recordSpkEvent" + n.first, VarLocation::HOST_DEVICE);
                backend.genVariableImplementation(runnerVarDecl, "uint32_t*", "recordSpkEvent" + n.first, VarLocation::HOST_DEVICE);
                backend.genVariableFree(runnerVarFree, "recordSpkEvent" + n.first, VarLocation::HOST_DEVICE);
            }
        }

        // If neuron group has axonal delays
        if (n.second.isDelayRequired()) {
            genHostDeviceScalar(backend, definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                "unsigned int", "spkQuePtr" + n.first, "0", mem);
        }

        // If neuron group needs to record its spike times
        if (n.second.isSpikeTimeRequired()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getTimePrecision(), "sT" + n.first, n.second.getSpikeTimeLocation(),
                             numNeuronDelaySlots, mem);

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeTimeLocation(),
                                backend.getPreferences().automaticCopy, n.first + "SpikeTimes",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, model.getTimePrecision(),
                                                                "sT" + n.first, n.second.getSpikeTimeLocation(), true, 
                                                                numNeuronDelaySlots);
                                });
        }

        // If neuron group needs to record its previous spike times
        if (n.second.isPrevSpikeTimeRequired()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getTimePrecision(), "prevST" + n.first, n.second.getPrevSpikeTimeLocation(),
                             numNeuronDelaySlots, mem);

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getPrevSpikeTimeLocation(),
                                backend.getPreferences().automaticCopy, n.first + "PreviousSpikeTimes",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, model.getTimePrecision(),
                                                                "prevST" + n.first, n.second.getPrevSpikeTimeLocation(), true,
                                                                numNeuronDelaySlots);
                                });
        }

        // If neuron group needs to record its spike-like-event times
        if (n.second.isSpikeEventTimeRequired()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getTimePrecision(), "seT" + n.first, n.second.getSpikeEventTimeLocation(),
                             numNeuronDelaySlots, mem);

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeTimeLocation(),
                                backend.getPreferences().automaticCopy, n.first + "SpikeEventTimes",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, model.getTimePrecision(),
                                                                "seT" + n.first, n.second.getSpikeEventTimeLocation(), true, 
                                                                numNeuronDelaySlots);
                                });
        }

        // If neuron group needs to record its previous spike-like-event times
        if (n.second.isPrevSpikeEventTimeRequired()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getTimePrecision(), "prevSET" + n.first, n.second.getPrevSpikeEventTimeLocation(),
                             numNeuronDelaySlots, mem);

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getPrevSpikeEventTimeLocation(),
                                backend.getPreferences().automaticCopy, n.first + "PreviousSpikeEventTimes",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, model.getTimePrecision(),
                                                                "prevSET" + n.first, n.second.getPrevSpikeEventTimeLocation(), true, 
                                                                numNeuronDelaySlots);
                                });
        }

        // If neuron group needs per-neuron RNGs
        if(n.second.isSimRNGRequired()) {
            backend.genPopulationRNG(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     "rng" + n.first, batchSize * n.second.getNumNeurons(), mem);
        }

        // Neuron state variables
        const auto neuronModel = n.second.getNeuronModel();
        const auto vars = neuronModel->getVars();
        std::vector<std::string> neuronStatePushPullFunctions;
        for(size_t i = 0; i < vars.size(); i++) {
            const auto *varInitSnippet = n.second.getVarInitialisers()[i].getSnippet();
            const unsigned int numCopies = getNumCopies(vars[i].access, batchSize);
            const size_t count = n.second.isVarQueueRequired(i) ? numCopies * n.second.getNumNeurons() * n.second.getNumDelaySlots() : numCopies * n.second.getNumNeurons();
            const bool autoInitialized = !varInitSnippet->getCode().empty();
            genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                        runnerPushFunc, runnerPullFunc, vars[i].type, vars[i].name + n.first,
                        n.second.getVarLocation(i), autoInitialized, count, mem, neuronStatePushPullFunctions);

            // Current variable push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getVarLocation(i),
                                backend.getPreferences().automaticCopy, "Current" + vars[i].name + n.first,
                                [&]()
                                {
                                    backend.genCurrentVariablePushPull(runnerPushFunc, runnerPullFunc, n.second, vars[i].type,
                                                                       vars[i].name, n.second.getVarLocation(i), numCopies);
                                });

            // Write getter to get access to correct pointer
            const bool delayRequired = (n.second.isVarQueueRequired(i) &&  n.second.isDelayRequired());
            genVarGetterScope(definitionsFunc, runnerGetterFunc, n.second.getVarLocation(i),
                              "Current" + vars[i].name + n.first, vars[i].type + "*",
                              [&]()
                              {
                                  runnerGetterFunc << "return " << vars[i].name << n.first;
                                  if(delayRequired) {
                                      runnerGetterFunc << " + (spkQuePtr" << n.first << " * " << n.second.getNumNeurons() << ")";
                                      if(numCopies > 1) {
                                          runnerGetterFunc << " + (batch * " << (n.second.getNumNeurons() * n.second.getNumDelaySlots()) << ")";
                                      }
                                  }
                                  else if(numCopies > 1) {
                                      runnerGetterFunc << " + (batch * " << n.second.getNumNeurons() << ")";
                                  }
                                  runnerGetterFunc << ";" << std::endl;
                              });

            // Loop through EGPs required to initialize neuron variable
            const auto extraGlobalParams = varInitSnippet->getExtraGlobalParams();
            for(size_t e = 0; e < extraGlobalParams.size(); e++) {
                genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar, 
                                    runnerVarDecl, runnerExtraGlobalParamFunc, 
                                    extraGlobalParams[e].type, extraGlobalParams[e].name + vars[i].name + n.first,
                                    true, VarLocation::HOST_DEVICE);
            }
        }

        // Add helper function to push and pull entire neuron state
        if(!backend.getPreferences().automaticCopy) {
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, 
                             n.first, backend.getPreferences().generateEmptyStatePushPull, 
                             neuronStatePushPullFunctions, statePushPullFunctions);
        }

        const auto extraGlobalParams = neuronModel->getExtraGlobalParams();
        for(size_t i = 0; i < extraGlobalParams.size(); i++) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc, 
                                extraGlobalParams[i].type, extraGlobalParams[i].name + n.first,
                                true, n.second.getExtraGlobalParamLocation(i));
        }

        if(!n.second.getCurrentSources().empty()) {
            allVarStreams << "// current source variables" << std::endl;
        }
        for (auto const *cs : n.second.getCurrentSources()) {
            const auto csModel = cs->getCurrentSourceModel();
            const auto csVars = csModel->getVars();

            std::vector<std::string> currentSourceStatePushPullFunctions;
            for(size_t i = 0; i < csVars.size(); i++) {
                const auto *varInitSnippet = cs->getVarInitialisers()[i].getSnippet();
                const bool autoInitialized = !varInitSnippet->getCode().empty();
                genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                            runnerPushFunc, runnerPullFunc, csVars[i].type, csVars[i].name + cs->getName(), cs->getVarLocation(i),
                            autoInitialized, getNumCopies(csVars[i].access, batchSize) * n.second.getNumNeurons(), mem, currentSourceStatePushPullFunctions);

                // Loop through EGPs required to initialize current source variable
                const auto extraGlobalParams = varInitSnippet->getExtraGlobalParams();
                for(size_t e = 0; e < extraGlobalParams.size(); e++) {
                    genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                        runnerVarDecl, runnerExtraGlobalParamFunc, 
                                        extraGlobalParams[e].type, extraGlobalParams[e].name + vars[i].name + cs->getName(),
                                        true, VarLocation::HOST_DEVICE);
                }
            }

            // Add helper function to push and pull entire current source state
            if(!backend.getPreferences().automaticCopy) {
                genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, 
                                 cs->getName(), backend.getPreferences().generateEmptyStatePushPull, 
                                 currentSourceStatePushPullFunctions, statePushPullFunctions);
            }

            const auto csExtraGlobalParams = csModel->getExtraGlobalParams();
            for(size_t i = 0; i < csExtraGlobalParams.size(); i++) {
                genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                    runnerVarDecl, runnerExtraGlobalParamFunc, 
                                    csExtraGlobalParams[i].type, csExtraGlobalParams[i].name + cs->getName(),
                                    true, cs->getExtraGlobalParamLocation(i));
            }
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// custom update variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    genCustomUpdate(modelMerged, backend,
                    definitionsVar, definitionsFunc, definitionsInternalVar,
                    runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
                    runnerPushFunc, runnerPullFunc, model.getCustomUpdates(),
                    mem, statePushPullFunctions, [](const CustomUpdateInternal &c) { return c.getSize(); });

    genCustomUpdate(modelMerged, backend,
                    definitionsVar, definitionsFunc, definitionsInternalVar,
                    runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
                    runnerPushFunc, runnerPullFunc, model.getCustomWUUpdates(),
                    mem, statePushPullFunctions, 
                    [&backend](const CustomUpdateWUInternal &c) 
                    { 
                        return c.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(*c.getSynapseGroup()); 
                    });
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// postsynaptic variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &n : model.getNeuronGroups()) {
        // Loop through merged incoming synaptic populations
        // **NOTE** because of merging we need to loop through postsynaptic models in this
        for(const auto *sg : n.second.getMergedInSyn()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getPrecision(), "inSyn" + sg->getPSModelTargetName(), sg->getInSynLocation(),
                             sg->getTrgNeuronGroup()->getNumNeurons() * batchSize, mem);

            if (sg->isDendriticDelayRequired()) {
                backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 model.getPrecision(), "denDelay" + sg->getPSModelTargetName(), sg->getDendriticDelayLocation(),
                                 (size_t)sg->getMaxDendriticDelayTimesteps() * (size_t)sg->getTrgNeuronGroup()->getNumNeurons() * batchSize, mem);
                genHostDeviceScalar(backend, definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                    "unsigned int", "denDelayPtr" + sg->getPSModelTargetName(), "0", mem);
            }

            if (sg->getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                const auto psmVars = sg->getPSModel()->getVars();
                for(size_t v = 0; v < psmVars.size(); v++) {
                    backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     psmVars[v].type, psmVars[v].name + sg->getPSModelTargetName(), sg->getPSVarLocation(v),
                                     sg->getTrgNeuronGroup()->getNumNeurons() * getNumCopies(psmVars[v].access, batchSize), mem);

                    // Loop through EGPs required to initialize PSM variable
                    const auto extraGlobalParams = sg->getPSVarInitialisers()[v].getSnippet()->getExtraGlobalParams();
                    for(size_t e = 0; e < extraGlobalParams.size(); e++) {
                        genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                            runnerVarDecl, runnerExtraGlobalParamFunc, 
                                            extraGlobalParams[e].type, extraGlobalParams[e].name + psmVars[v].name + sg->getPSModelTargetName(),
                                            true, VarLocation::HOST_DEVICE);
                    }
                }
            }
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// synapse connectivity" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    std::vector<std::string> connectivityPushPullFunctions;
    for(const auto &s : model.getSynapseGroups()) {
        // If this synapse group isn't a weight sharing slave i.e. it's connectivity isn't initialized on the master
        if(!s.second.isWeightSharingSlave()) {
            const auto *snippet = s.second.getConnectivityInitialiser().getSnippet();
            const bool autoInitialized = !snippet->getRowBuildCode().empty() || !snippet->getColBuildCode().empty();

            if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
                const size_t gpSize = ceilDivide((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(s.second), 32);
                backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 "uint32_t", "gp" + s.second.getName(), s.second.getSparseConnectivityLocation(), gpSize, mem);

                // Generate push and pull functions for bitmask
                genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getSparseConnectivityLocation(),
                                    backend.getPreferences().automaticCopy, s.second.getName() + "Connectivity", connectivityPushPullFunctions,
                                    [&]()
                                    {
                                        // Row lengths
                                        backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, "uint32_t", "gp" + s.second.getName(),
                                                                    s.second.getSparseConnectivityLocation(), autoInitialized, gpSize);
                                    });
            }
            else if(s.second.getMatrixType() & SynapseMatrixConnectivity::SPARSE) {
                const VarLocation varLoc = s.second.getSparseConnectivityLocation();
                const size_t size = s.second.getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(s.second);

                // Maximum row length constant
                definitionsVar << "EXPORT_VAR const unsigned int maxRowLength" << s.second.getName() << ";" << std::endl;
                runnerVarDecl << "const unsigned int maxRowLength" << s.second.getName() << " = " << backend.getSynapticMatrixRowStride(s.second) << ";" << std::endl;

                // Row lengths
                backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 "unsigned int", "rowLength" + s.second.getName(), varLoc, s.second.getSrcNeuronGroup()->getNumNeurons(), mem);

                // Target indices
                backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 s.second.getSparseIndType(), "ind" + s.second.getName(), varLoc, size, mem);


                // If synapse remap structure is required, allocate synRemap
                // **THINK** this is over-allocating
                if(backend.isSynRemapRequired(s.second)) {
                    backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     "unsigned int", "synRemap" + s.second.getName(), VarLocation::DEVICE, size + 1, mem);
                }

                // **TODO** remap is not always required
                if(backend.isPostsynapticRemapRequired() && !s.second.getWUModel()->getLearnPostCode().empty()) {
                    const size_t postSize = (size_t)s.second.getTrgNeuronGroup()->getNumNeurons() * (size_t)s.second.getMaxSourceConnections();

                    // Allocate column lengths
                    backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     "unsigned int", "colLength" + s.second.getName(), VarLocation::DEVICE, s.second.getTrgNeuronGroup()->getNumNeurons(), mem);

                    // Allocate remap
                    backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     "unsigned int", "remap" + s.second.getName(), VarLocation::DEVICE, postSize, mem);
                }

                // Generate push and pull functions for sparse connectivity
                genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getSparseConnectivityLocation(),
                                    backend.getPreferences().automaticCopy, s.second.getName() + "Connectivity", connectivityPushPullFunctions,
                                    [&]()
                                    {
                                        // Row lengths
                                        backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, "unsigned int", "rowLength" + s.second.getName(), 
                                                                    s.second.getSparseConnectivityLocation(), autoInitialized, s.second.getSrcNeuronGroup()->getNumNeurons());

                                        // Target indices
                                        backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, "unsigned int", "ind" + s.second.getName(), 
                                                                    s.second.getSparseConnectivityLocation(), autoInitialized, size);
                                    });
            }
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// synapse variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &s : model.getSynapseGroups()) {
        const auto *wu = s.second.getWUModel();
        const auto *psm = s.second.getPSModel();

        // If group isn't a weight sharing slave and per-synapse variables should be individual
        const bool individualWeights = (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL);
        const bool proceduralWeights = (s.second.getMatrixType() & SynapseMatrixWeight::PROCEDURAL);
        std::vector<std::string> synapseGroupStatePushPullFunctions;
        if (!s.second.isWeightSharingSlave() && (individualWeights || proceduralWeights)) {
            const size_t size = (size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * (size_t)backend.getSynapticMatrixRowStride(s.second);

            const auto wuVars = wu->getVars();
            for(size_t i = 0; i < wuVars.size(); i++) {
                const auto *varInitSnippet = s.second.getWUVarInitialisers()[i].getSnippet();
                if(individualWeights) {
                    const bool autoInitialized = !varInitSnippet->getCode().empty();
                    genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                runnerPushFunc, runnerPullFunc, wuVars[i].type, wuVars[i].name + s.second.getName(), s.second.getWUVarLocation(i),
                                autoInitialized, size * getNumCopies(wuVars[i].access, batchSize), mem, synapseGroupStatePushPullFunctions);
                }

                // Loop through EGPs required to initialize WUM variable
                const auto extraGlobalParams = varInitSnippet->getExtraGlobalParams();
                for(size_t e = 0; e < extraGlobalParams.size(); e++) {
                    genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                        runnerVarDecl, runnerExtraGlobalParamFunc, 
                                        extraGlobalParams[e].type, extraGlobalParams[e].name + wuVars[i].name + s.second.getName(),
                                        true, VarLocation::HOST_DEVICE);
                }
            }
        }

        // Presynaptic W.U.M. variables
        const size_t preSize = (s.second.getDelaySteps() == NO_DELAY)
                ? s.second.getSrcNeuronGroup()->getNumNeurons()
                : s.second.getSrcNeuronGroup()->getNumNeurons() * s.second.getSrcNeuronGroup()->getNumDelaySlots();
        const auto wuPreVars = wu->getPreVars();
        for(size_t i = 0; i < wuPreVars.size(); i++) {
            const auto *varInitSnippet = s.second.getWUPreVarInitialisers()[i].getSnippet();
            const bool autoInitialized = !varInitSnippet->getCode().empty();
            genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                        runnerPushFunc, runnerPullFunc, wuPreVars[i].type, wuPreVars[i].name + s.second.getName(),
                        s.second.getWUPreVarLocation(i), autoInitialized, preSize * getNumCopies(wuPreVars[i].access, batchSize), mem, synapseGroupStatePushPullFunctions);

            // Loop through EGPs required to initialize WUM variable
            const auto extraGlobalParams = varInitSnippet->getExtraGlobalParams();
            for(size_t e = 0; e < extraGlobalParams.size(); e++) {
                genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                    runnerVarDecl, runnerExtraGlobalParamFunc, 
                                    extraGlobalParams[e].type, extraGlobalParams[e].name + wuPreVars[i].name + s.second.getName(),
                                    true, VarLocation::HOST_DEVICE);
            }
        }

        // Postsynaptic W.U.M. variables
        const size_t postSize = (s.second.getBackPropDelaySteps() == NO_DELAY)
                ? s.second.getTrgNeuronGroup()->getNumNeurons()
                : s.second.getTrgNeuronGroup()->getNumNeurons() * s.second.getTrgNeuronGroup()->getNumDelaySlots();
        const auto wuPostVars = wu->getPostVars();
        for(size_t i = 0; i < wuPostVars.size(); i++) {
            const auto *varInitSnippet = s.second.getWUPostVarInitialisers()[i].getSnippet();
            const bool autoInitialized = !varInitSnippet->getCode().empty();
            genVariable(backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                        runnerPushFunc, runnerPullFunc, wuPostVars[i].type, wuPostVars[i].name + s.second.getName(), s.second.getWUPostVarLocation(i),
                        autoInitialized, postSize * getNumCopies(wuPostVars[i].access, batchSize), mem, synapseGroupStatePushPullFunctions);

            // Loop through EGPs required to initialize WUM variable
            const auto extraGlobalParams = varInitSnippet->getExtraGlobalParams();
            for(size_t e = 0; e < extraGlobalParams.size(); e++) {
                genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                    runnerVarDecl, runnerExtraGlobalParamFunc, 
                                    extraGlobalParams[e].type, extraGlobalParams[e].name + wuPostVars[i].name + s.second.getName(),
                                    true, VarLocation::HOST_DEVICE);
            }
        }

        // If this synapse group's postsynaptic models hasn't been merged (which makes pulling them somewhat ambiguous)
        // **NOTE** we generated initialisation and declaration code earlier - here we just generate push and pull as we want this per-synapse group
        if(!s.second.isPSModelMerged()) {
            // Add code to push and pull inSyn
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getInSynLocation(),
                                backend.getPreferences().automaticCopy, "inSyn" + s.second.getName(), synapseGroupStatePushPullFunctions,
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, model.getPrecision(), "inSyn" + s.second.getName(), s.second.getInSynLocation(),
                                                                true, s.second.getTrgNeuronGroup()->getNumNeurons() * batchSize);
                                });

            // If this synapse group has individual postsynaptic model variables
            if (s.second.getMatrixType() & SynapseMatrixWeight::INDIVIDUAL_PSM) {
                const auto psmVars = psm->getVars();
                for(size_t i = 0; i < psmVars.size(); i++) {
                    const bool autoInitialized = !s.second.getPSVarInitialisers()[i].getSnippet()->getCode().empty();
                    genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getPSVarLocation(i),
                                        backend.getPreferences().automaticCopy, psmVars[i].name + s.second.getName(), synapseGroupStatePushPullFunctions,
                                        [&]()
                                        {
                                            backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, psmVars[i].type, psmVars[i].name + s.second.getName(), s.second.getPSVarLocation(i),
                                                                        autoInitialized, s.second.getTrgNeuronGroup()->getNumNeurons() * getNumCopies(psmVars[i].access, batchSize));
                                        });
                }
            }
        }

        // Add helper function to push and pull entire synapse group state
        if(!backend.getPreferences().automaticCopy) {
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, 
                             s.second.getName(), backend.getPreferences().generateEmptyStatePushPull, 
                             synapseGroupStatePushPullFunctions, statePushPullFunctions);
        }

        const auto psmExtraGlobalParams = psm->getExtraGlobalParams();
        for(size_t i = 0; i < psmExtraGlobalParams.size(); i++) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc, 
                                psmExtraGlobalParams[i].type, psmExtraGlobalParams[i].name + s.second.getName(),
                                true, s.second.getPSExtraGlobalParamLocation(i));
        }

        const auto wuExtraGlobalParams = wu->getExtraGlobalParams();
        for(size_t i = 0; i < wuExtraGlobalParams.size(); i++) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc, 
                                wuExtraGlobalParams[i].type, wuExtraGlobalParams[i].name + s.second.getName(),
                                true, s.second.getWUExtraGlobalParamLocation(i));
        }

        // If group isn't a weight sharing slave 
        if(!s.second.isWeightSharingSlave()) {
            const auto sparseConnExtraGlobalParams = s.second.getConnectivityInitialiser().getSnippet()->getExtraGlobalParams();
            for(size_t i = 0; i < sparseConnExtraGlobalParams.size(); i++) {
                genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                    runnerVarDecl, runnerExtraGlobalParamFunc, 
                                    sparseConnExtraGlobalParams[i].type, sparseConnExtraGlobalParams[i].name + s.second.getName(),
                                    s.second.getConnectivityInitialiser().getSnippet()->getHostInitCode().empty(),
                                    s.second.getSparseConnectivityExtraGlobalParamLocation(i));
            }
        }
    }
    allVarStreams << std::endl;

    // End extern C block around variable declarations
    runnerVarDecl << "}  // extern \"C\"" << std::endl;
 
    // Write pre-amble to runner
    backend.genRunnerPreamble(runner, modelMerged, mem);

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

    if(!backend.getPreferences().automaticCopy) {
        // ---------------------------------------------------------------------
        // Function for copying all state to device
        runner << "void copyStateToDevice(bool uninitialisedOnly)";
        {
            CodeStream::Scope b(runner);
            for(const auto &g : statePushPullFunctions) {
                runner << "push" << g << "StateToDevice(uninitialisedOnly);" << std::endl;
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
            for(const auto &g : statePushPullFunctions) {
                runner << "pull" << g << "StateFromDevice();" << std::endl;
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
    }

    // If model uses recording
    if(model.isRecordingInUse()) {
        runner << "void allocateRecordingBuffers(unsigned int timesteps)";
        {
            CodeStream::Scope b(runner);

            // Cache number of recording timesteps in global variable
            runner << "numRecordingTimesteps = timesteps;" << std::endl;

            // Loop through neuron groups
            for(const auto &n : model.getNeuronGroups()) {
                CodeStream::Scope b(runner);

                // Calculate number of words required for spike/spike event buffers
                if(n.second.isSpikeRecordingEnabled() || n.second.isSpikeEventRecordingEnabled()) {
                    runner << "const unsigned int numWords = " << (ceilDivide(n.second.getNumNeurons(), 32) * model.getBatchSize()) << " * timesteps;" << std::endl;
                }

                // Allocate spike array if required
                // **YUCK** maybe this should be renamed genDynamicArray
                if(n.second.isSpikeRecordingEnabled()) {
                    CodeStream::Scope b(runner);
                    backend.genExtraGlobalParamAllocation(runner, "uint32_t*", "recordSpk" + n.first, VarLocation::HOST_DEVICE, "numWords");

                    // Get destinations in merged structures, this EGP 
                    // needs to be copied to and call push function
                    const auto &mergedDestinations = modelMerged.getMergedEGPDestinations("recordSpk" + n.first, backend);
                    for(const auto &v : mergedDestinations) {
                        runner << "pushMerged" << v.first << v.second.mergedGroupIndex << v.second.fieldName << "ToDevice(";
                        runner << v.second.groupIndex << ", " << backend.getDeviceVarPrefix() << "recordSpk" + n.first << ");" << std::endl;
                    }
                }

                // Allocate spike event array if required
                // **YUCK** maybe this should be renamed genDynamicArray
                if(n.second.isSpikeEventRecordingEnabled()) {
                    CodeStream::Scope b(runner);
                    backend.genExtraGlobalParamAllocation(runner, "uint32_t*", "recordSpkEvent" + n.first, VarLocation::HOST_DEVICE, "numWords");

                    // Get destinations in merged structures, this EGP 
                    // needs to be copied to and call push function
                    const auto &mergedDestinations = modelMerged.getMergedEGPDestinations("recordSpkEvent" + n.first, backend);
                    for(const auto &v : mergedDestinations) {
                        runner << "pushMerged" << v.first << v.second.mergedGroupIndex << v.second.fieldName << "ToDevice(";
                        runner << v.second.groupIndex << ", " << backend.getDeviceVarPrefix() << "recordSpkEvent" + n.first << ");" << std::endl;
                    }
                }
            }
        }
        runner << std::endl;

        runner << "void pullRecordingBuffersFromDevice()";
        {
            CodeStream::Scope b(runner);
            
            // Check recording buffer has been allocated
            runner << "if(numRecordingTimesteps == 0)";
            {
                CodeStream::Scope b(runner);
                runner << "throw std::runtime_error(\"Recording buffer not allocated - cannot pull from device\");" << std::endl;
            }

            // Loop through neuron groups
            // **THINK** could use asynchronous copies and sync on last one
            for(const auto &n : model.getNeuronGroups()) {
                CodeStream::Scope b(runner);

                // Calculate number of words required for spike/spike event buffers
                if(n.second.isSpikeRecordingEnabled() || n.second.isSpikeEventRecordingEnabled()) {
                    runner << "const unsigned int numWords = " << (ceilDivide(n.second.getNumNeurons(), 32) * model.getBatchSize()) << " * numRecordingTimesteps;" << std::endl;
                }

                // Pull spike array if required
                // **YUCK** maybe this should be renamed pullDynamicArray
                if(n.second.isSpikeRecordingEnabled()) {
                    CodeStream::Scope b(runner);
                    backend.genExtraGlobalParamPull(runner, "uint32_t*", "recordSpk" + n.first, VarLocation::HOST_DEVICE, "numWords");
                }
                // AllocaPullte spike event array if required
                // **YUCK** maybe this should be renamed pullDynamicArray
                if(n.second.isSpikeEventRecordingEnabled()) {
                    CodeStream::Scope b(runner);
                    backend.genExtraGlobalParamPull(runner, "uint32_t*", "recordSpkEvent" + n.first, VarLocation::HOST_DEVICE, "numWords");
                }
            }
        }
        runner << std::endl;
    }

    // ---------------------------------------------------------------------
    // Function for setting the device and the host's global variables.
    // Also estimates memory usage on device ...
    runner << "void allocateMem(" << backend.getAllocateMemParams(modelMerged) << ")";
    {
        CodeStream::Scope b(runner);

        // Generate preamble -this is the first bit of generated code called by user simulations
        // so global initialisation is often performed here
        backend.genAllocateMemPreamble(runner, modelMerged, mem);

        // Write variable allocations to runner
        runner << runnerVarAllocStream.str();

        // Write merged struct allocations to runner
        runner << runnerMergedStructAllocStream.str();
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
    // Function to return amount of free device memory in bytes
    runner << "size_t getFreeDeviceMemBytes()";
    {
        CodeStream::Scope b(runner);

        // Generate code to return free memory
        backend.genReturnFreeDeviceMemoryBytes(runner);
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
   
        for(const auto &n : model.getNeuronGroups()) {
            if (n.second.isDelayRequired()) {
                runner << "spkQuePtr" << n.first << " = (spkQuePtr" << n.first << " + 1) % " << n.second.getNumDelaySlots() << ";" << std::endl;
            }
        }

        // Update neuronal state
        runner << "updateNeurons(t";
        if(model.isRecordingInUse()) {
            runner << ", (unsigned int)(iT % numRecordingTimesteps)";
        }
        runner << "); " << std::endl;

        // Generate code to advance host side dendritic delay buffers
        for(const auto &n : model.getNeuronGroups()) {
            // Loop through incoming synaptic populations
            for(const auto *sg : n.second.getMergedInSyn()) {
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
    if(!backend.getPreferences().automaticCopy) {
        definitions << "EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);" << std::endl;
        definitions << "EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);" << std::endl;
        definitions << "EXPORT_FUNC void copyStateFromDevice();" << std::endl;
        definitions << "EXPORT_FUNC void copyCurrentSpikesFromDevice();" << std::endl;
        definitions << "EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();" << std::endl;
    }

    if(model.isRecordingInUse()) {
        definitions << "EXPORT_FUNC void allocateRecordingBuffers(unsigned int timesteps);" << std::endl;
        definitions << "EXPORT_FUNC void pullRecordingBuffersFromDevice();" << std::endl;
    }
    definitions << "EXPORT_FUNC void allocateMem(" << backend.getAllocateMemParams(modelMerged) << ");" << std::endl;
    definitions << "EXPORT_FUNC void freeMem();" << std::endl;
    definitions << "EXPORT_FUNC size_t getFreeDeviceMemBytes();" << std::endl;
    definitions << "EXPORT_FUNC void stepTime();" << std::endl;
    definitions << std::endl;
    definitions << "// Functions generated by backend" << std::endl;
    definitions << "EXPORT_FUNC void updateNeurons(" << model.getTimePrecision() << " t";
    if(model.isRecordingInUse()) {
        definitions << ", unsigned int recordingTimestep";
    }
    definitions << "); " << std::endl;
    definitions << "EXPORT_FUNC void updateSynapses(" << model.getTimePrecision() << " t);" << std::endl;
    definitions << "EXPORT_FUNC void initialize();" << std::endl;
    definitions << "EXPORT_FUNC void initializeSparse();" << std::endl;
    
    // Generate function definitions for each custom update
    for(const auto &g : customUpdateGroups) {
        definitions << "EXPORT_FUNC void update" << g << "();" << std::endl;
    }
#ifdef MPI_ENABLE
    definitions << "// MPI functions" << std::endl;
    definitions << "EXPORT_FUNC void generateMPI();" << std::endl;
#endif

    // End extern C block around definitions
    definitions << "}  // extern \"C\"" << std::endl;
    definitionsInternal << "}  // extern \"C\"" << std::endl;

    return mem;
}
