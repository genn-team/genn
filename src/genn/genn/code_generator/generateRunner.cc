#include "code_generator/generateRunner.h"

// Standard C++ includes
#include <fstream>
#include <numeric>
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

using namespace GeNN;
using namespace GeNN::CodeGenerator;

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
unsigned int getNumVarCopies(VarAccess varAccess, unsigned int batchSize, bool batched = true)
{
    return ((varAccess & VarAccessDuplication::SHARED) || !batched) ? 1 : batchSize;
}
//--------------------------------------------------------------------------
unsigned int getNumVarElements(VarAccess varAccess, unsigned int numNeurons)
{
    return (varAccess & VarAccessDuplication::SHARED_NEURON) ? 1 : numNeurons;
}
//--------------------------------------------------------------------------
unsigned int getVarSize(VarAccess varAccess, unsigned int numElements, unsigned int batchSize, 
                        unsigned int delaySlots = 1, bool batched = true)
{
    return getNumVarCopies(varAccess, batchSize, batched) * getNumVarElements(varAccess, numElements) * delaySlots;
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
template<typename T>
void genHostScalar(CodeStream &definitionsVar, CodeStream &runnerVarDecl,
                   const std::string &name, const std::string &value)
{
    definitionsVar << "EXPORT_VAR " << T::getInstance()->getName() << " " << name << ";" << std::endl;
    runnerVarDecl << T::getInstance()->getName() << " " << name << " = " << value << ";" << std::endl;
}
//--------------------------------------------------------------------------
template<typename T>
void genHostDeviceScalar(const ModelSpecMerged &modelMerged, const BackendBase &backend, CodeStream &definitionsVar, 
                         CodeStream &definitionsInternalVar, CodeStream &runnerVarDecl, CodeStream &runnerVarAlloc, CodeStream &runnerVarFree,
                         const std::string &name, const std::string &hostValue, MemAlloc &mem)
{
    // Generate a host scalar
    genHostScalar<T>(definitionsVar, runnerVarDecl, name, hostValue);

    // Generate a single-element array on device
    if(backend.isDeviceScalarRequired()) {
        backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         T::getInstance(), modelMerged.getTypeContext(), name, VarLocation::DEVICE, 1, mem);
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
                       const std::string &typeName, std::function<void()> handler)
{
    // If this variable has a location that allows pushing and pulling and hence getting a host pointer
    if(canPushPullVar(loc)) {
        // Export getter
        definitionsFunc << "EXPORT_FUNC " << typeName << " get" << description << "(unsigned int batch = 0); " << std::endl;

        // Define getter
        runnerGetterFunc << typeName << " get" << description << "(" << "unsigned int batch" << ")";
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
void genVariable(const ModelSpecMerged &modelMerged, const BackendBase &backend,
                 CodeStream &definitionsVar, CodeStream &definitionsFunc, CodeStream &definitionsInternal, 
                 CodeStream &runner, CodeStream &allocations, CodeStream &free, CodeStream &push, CodeStream &pull, 
                 const Type::ValueBase *type, const std::string &name,
                 VarLocation loc, bool autoInitialized, size_t count, MemAlloc &mem,
                 std::vector<std::string> &statePushPullFunction)
{
    // Generate push and pull functions
    genVarPushPullScope(definitionsFunc, push, pull, loc, backend.getPreferences().automaticCopy, name, statePushPullFunction,
        [&]()
        {
            backend.genVariablePushPull(push, pull, type, modelMerged.getTypeContext(), name, loc, autoInitialized, count);
        });

    // Generate variables
    backend.genArray(definitionsVar, definitionsInternal, runner, allocations, free,
                     type, modelMerged.getTypeContext(), name, loc, count, mem);
}
//-------------------------------------------------------------------------
void genExtraGlobalParam(const ModelSpecMerged &modelMerged, const BackendBase &backend, CodeStream &definitionsVar,
                         CodeStream &definitionsFunc, CodeStream &definitionsInternalVar, CodeStream &runner,
                         CodeStream &extraGlobalParam, const Type::NumericBase *type, const std::string &name, bool apiRequired, VarLocation loc)
{
    // Generate variables
    backend.genVariableDefinition(definitionsVar, definitionsInternalVar, type, modelMerged.getTypeContext(), name, loc);
    backend.genVariableInstantiation(runner, type, modelMerged.getTypeContext(), name, loc);

    // If API is required
    if(apiRequired) {
        // Write definitions for functions to allocate and free extra global param
        definitionsFunc << "EXPORT_FUNC void allocate" << name << "(unsigned int count);" << std::endl;
        definitionsFunc << "EXPORT_FUNC void free" << name << "();" << std::endl;

        // Write allocation function
        extraGlobalParam << "void allocate" << name << "(unsigned int count)";
        {
            CodeStream::Scope a(extraGlobalParam);
            backend.genVariableDynamicAllocation(extraGlobalParam, type, modelMerged.getTypeContext(), name, loc);

            // Loop through destinations in merged structures, the device EGP needs to be copied to
            // **TODO** rename to dynamic
            if(modelMerged.anyMergedEGPDestinations(backend.getDeviceVarPrefix() + name)) {
                const auto &mergedDestinations = modelMerged.getMergedEGPDestinations(backend.getDeviceVarPrefix() + name);
                for (const auto &v : mergedDestinations) {
                    // If this is a host group, directly update merged group data structure
                    if (v.second.hostGroup) {
                        extraGlobalParam << "merged" << v.first << "Group" << v.second.mergedGroupIndex << "[" << v.second.groupIndex << "]";
                        extraGlobalParam << "." << v.second.fieldName << " = " << backend.getDeviceVarPrefix() << name << ";" << std::endl;
                    }
                    // Otherwise, call push function which 
                    else {
                        extraGlobalParam << "pushMerged" << v.first << v.second.mergedGroupIndex << v.second.fieldName << "ToDevice(";
                        extraGlobalParam << v.second.groupIndex << ", " << backend.getDeviceVarPrefix() << name << ");" << std::endl;
                    }
                }
            }
            
            // If backend has a device variable prefix (otherwise, previous loop will have exactly the same effect)
            if(!backend.getDeviceVarPrefix().empty() && modelMerged.anyMergedEGPDestinations(name)) {
                // Loop through destinations in merged structures, the host EGP needs to be copied to
                const auto &mergedDestinations = modelMerged.getMergedEGPDestinations(name);
                for(const auto &v : mergedDestinations) {
                    // Assert that host variables are only being mapped to host groups
                    assert(v.second.hostGroup);
                    
                    extraGlobalParam << "merged" << v.first << "Group" << v.second.mergedGroupIndex << "[" << v.second.groupIndex << "]";
                    extraGlobalParam << "." << v.second.fieldName << " = " << name << ";" << std::endl;
                }   
            }
            
            // If backend has a host variable prefix
            if(!backend.getHostVarPrefix().empty() && modelMerged.anyMergedEGPDestinations(backend.getHostVarPrefix() + name)) {
                // Loop through destinations in merged structures, the host EGP needs to be copied to
                const auto &mergedDestinations = modelMerged.getMergedEGPDestinations(backend.getHostVarPrefix() + name);
                for(const auto &v : mergedDestinations) {
                    // Assert that host variables are only being mapped to host groups
                    assert(v.second.hostGroup);
                    
                    extraGlobalParam << "merged" << v.first << "Group" << v.second.mergedGroupIndex << "[" << v.second.groupIndex << "]";
                    extraGlobalParam << "." << v.second.fieldName << " = " << backend.getHostVarPrefix() <<name << ";" << std::endl;
                }   
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
                backend.genVariableDynamicPush(extraGlobalParam, type, modelMerged.getTypeContext(), name, loc);
            }

            if(backend.getPreferences().generateExtraGlobalParamPull) {
                // Write definitions for pull functions
                definitionsFunc << "EXPORT_FUNC void pull" << name << "FromDevice(unsigned int count);" << std::endl;

                // Write pull function
                extraGlobalParam << "void pull" << name << "FromDevice(unsigned int count)";
                {
                    CodeGenerator::CodeStream::Scope a(extraGlobalParam);
                    backend.genVariableDynamicPull(extraGlobalParam, type, modelMerged.getTypeContext(), name, loc);
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
template<typename V, typename G, typename S>
void genRunnerVars(const ModelSpecMerged &modelMerged, const BackendBase &backend, 
                   CodeStream &definitionsVar, CodeStream &definitionsFunc, CodeStream &definitionsInternalVar,
                   CodeStream &runnerVarDecl, CodeStream &runnerVarAlloc, CodeStream &runnerVarFree, 
                   CodeStream &runnerExtraGlobalParamFunc, CodeStream &runnerPushFunc, CodeStream &runnerPullFunc, 
                   const G &group, MemAlloc &mem, std::vector<std::string> &statePushPullFunctions, S getSizeFn)
{
    // Loop through variables
    const V varAdaptor(group);
    for(const auto &var : varAdaptor.getVars()) {
        const auto *varInitSnippet = varAdaptor.getVarInitialisers().at(var.name).getSnippet();
        const bool autoInitialized = !varInitSnippet->getCode().empty();
        genVariable(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                    runnerPushFunc, runnerPullFunc, var.type, var.name + group.getName(), varAdaptor.getVarLocation(var.name),
                    autoInitialized, getSizeFn(group, var), mem, statePushPullFunctions);

        // Loop through EGPs required to initialize variable
        for(const auto &egp : varInitSnippet->getExtraGlobalParams()) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc,
                                egp.type, egp.name + var.name + group.getName(),
                                true, VarLocation::HOST_DEVICE);
        }
    }
}
//-------------------------------------------------------------------------
template<typename V, typename G, typename S>
void genRunnerFusedVars(const ModelSpecMerged &modelMerged, const BackendBase &backend, 
                        CodeStream &definitionsVar, CodeStream &definitionsFunc, CodeStream &definitionsInternalVar,
                        CodeStream &runnerVarDecl, CodeStream &runnerVarAlloc, CodeStream &runnerVarFree, 
                        CodeStream &runnerExtraGlobalParamFunc, const G &group, MemAlloc &mem, S getSizeFn)
{
    // Loop through variables
    const V varAdaptor(group);
    for(const auto &var : varAdaptor.getVars()) {
        backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         var.type, modelMerged.getTypeContext(), var.name + varAdaptor.getFusedVarSuffix(), varAdaptor.getVarLocation(var.name),
                         getSizeFn(group, var), mem);

        // Loop through EGPs required to initialize variable
        for(const auto &egp : varAdaptor.getVarInitialisers().at(var.name).getSnippet()->getExtraGlobalParams()) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc, 
                                egp.type, egp.name + var.name + varAdaptor.getFusedVarSuffix(),
                                true, VarLocation::HOST_DEVICE);
        }
    }
}
//-------------------------------------------------------------------------
template<typename V, typename G, typename S>
void genRunnerFusedVarPushPull(const ModelSpecMerged &modelMerged, const BackendBase &backend, CodeStream &definitionsFunc, 
                               CodeStream &runnerPushFunc, CodeStream &runnerPullFunc, const G &group, 
                               std::vector<std::string> &groupStatePushPullFunctions, S getSizeFn)
{
    // Loop through variables
    const V varAdaptor(group);
    for(const auto &var : varAdaptor.getVars()) {
        const bool autoInitialized = !varAdaptor.getVarInitialisers().at(var.name).getSnippet()->getCode().empty();
        genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, varAdaptor.getVarLocation(var.name),
                            backend.getPreferences().automaticCopy, var.name + group.getName(), groupStatePushPullFunctions,
                            [&]()
                            {
                                backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, 
                                                            var.type, modelMerged.getTypeContext(), var.name + group.getName(), 
                                                            varAdaptor.getVarLocation(var.name), autoInitialized, getSizeFn(group, var));
                            });
    }
}
//-------------------------------------------------------------------------
template<typename E, typename G>
void genRunnerEGPs(const ModelSpecMerged &modelMerged, const BackendBase &backend,
                   CodeStream &definitionsVar, CodeStream &definitionsFunc, CodeStream &definitionsInternalVar,
                   CodeStream &runnerVarDecl, CodeStream &runnerExtraGlobalParamFunc, const G &group)
{
    // Loop through EGPs
    const E egpAdaptor(group);
    for(const auto &egp: egpAdaptor.getEGPs()) {
        genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                            runnerVarDecl, runnerExtraGlobalParamFunc,
                            egp.type, egp.name + group.getName(),
                            true, egpAdaptor.getEGPLocation(egp.name));
    }
}
//-------------------------------------------------------------------------
template<typename V, typename E, typename C, typename S>
void genCustomUpdate(const ModelSpecMerged &modelMerged, const BackendBase &backend, 
                     CodeStream &definitionsVar, CodeStream &definitionsFunc, CodeStream &definitionsInternalVar,
                     CodeStream &runnerVarDecl, CodeStream &runnerVarAlloc, CodeStream &runnerVarFree, CodeStream &runnerExtraGlobalParamFunc,
                     CodeStream &runnerPushFunc, CodeStream &runnerPullFunc, const std::map<std::string, C> &customUpdates,
                     MemAlloc &mem, std::vector<std::string> &statePushPullFunctions, S getSizeFn)
{
    // Loop through custom updates
    for(const auto &c : customUpdates) {
        // Generate variables
        std::vector<std::string> groupStatePushPullFunctions;
        genRunnerVars<V>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                         runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
                         runnerPushFunc, runnerPullFunc, c.second, mem, groupStatePushPullFunctions, getSizeFn);

        // Generate EGPs
        genRunnerEGPs<E>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                         runnerVarDecl, runnerExtraGlobalParamFunc, c.second);

        // Add helper function to push and pull entire group state
        if(!backend.getPreferences().automaticCopy) {
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc,
                             c.first, backend.getPreferences().generateEmptyStatePushPull,
                             groupStatePushPullFunctions, statePushPullFunctions);
        }
    }
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::CodeGenerator
//--------------------------------------------------------------------------
MemAlloc GeNN::CodeGenerator::generateRunner(const filesystem::path &outputPath, const ModelSpecMerged &modelMerged, 
                                             const BackendBase &backend, const std::string &suffix)
{
    // Create output streams to write to file and wrap in CodeStreams
    std::ofstream definitionsStream((outputPath / ("definitions" + suffix + ".h")).str());
    std::ofstream definitionsInternalStream((outputPath / ("definitionsInternal" + suffix + ".h")).str());
    std::ofstream runnerStream((outputPath / ("runner" + suffix + ".cc")).str());
    CodeStream definitions(definitionsStream);
    CodeStream definitionsInternal(definitionsInternalStream);
    CodeStream runner(runnerStream);

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
    definitionsInternal << "#include \"definitions" << suffix << ".h\"" << std::endl << std::endl;
    backend.genDefinitionsInternalPreamble(definitionsInternal, modelMerged);
    
    // write DT macro
    const ModelSpecInternal &model = modelMerged.getModel();
    definitions << "#define DT " << Utils::writePreciseString(model.getDT()) << model.getTimePrecision()->getLiteralSuffix(modelMerged.getTypeContext()) << std::endl;
    
    // Typedefine scalar type
    definitions << "typedef " << model.getPrecision()->getName() << " scalar;" << std::endl;

    // Write ranges of scalar and time types
    genTypeRange(definitions, model.getPrecision(), modelMerged.getTypeContext(), "SCALAR");
    genTypeRange(definitions, model.getTimePrecision(), modelMerged.getTypeContext(), "TIME");

    definitions << "// ------------------------------------------------------------------------" << std::endl;
    definitions << "// bit tool macros" << std::endl;
    definitions << "#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x" << std::endl;
    definitions << "#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1" << std::endl;
    definitions << "#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0" << std::endl;
    definitions << std::endl;

    // Write runner preamble
    runner << "#include \"definitionsInternal" << suffix << ".h\"" << std::endl << std::endl;

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
    definitionsVar << "EXPORT_VAR " << model.getTimePrecision()->getName() << " t;" << std::endl;
    runnerVarDecl << "unsigned long long iT;" << std::endl;
    runnerVarDecl << model.getTimePrecision()->getName() << " t;" << std::endl;

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
    std::transform(model.getCustomConnectivityUpdates().cbegin(), model.getCustomConnectivityUpdates().cend(),
                   std::inserter(customUpdateGroups, customUpdateGroups.end()),
                   [](const ModelSpec::CustomConnectivityUpdateValueType &v) { return v.second.getUpdateGroupName(); });

    // Generate variables to store total elapsed time
    // **NOTE** we ALWAYS generate these so usercode doesn't require #ifdefs around timing code
    genHostScalar<Type::Double>(definitionsVar, runnerVarDecl, "initTime", "0.0");
    genHostScalar<Type::Double>(definitionsVar, runnerVarDecl, "initSparseTime", "0.0");
    genHostScalar<Type::Double>(definitionsVar, runnerVarDecl, "neuronUpdateTime", "0.0");
    genHostScalar<Type::Double>(definitionsVar, runnerVarDecl, "presynapticUpdateTime", "0.0");
    genHostScalar<Type::Double>(definitionsVar, runnerVarDecl, "postsynapticUpdateTime", "0.0");
    genHostScalar<Type::Double>(definitionsVar, runnerVarDecl, "synapseDynamicsTime", "0.0");

    // Generate variables to store total elapsed time for each custom update group
    for(const auto &g : customUpdateGroups) {
        genHostScalar<Type::Double>(definitionsVar, runnerVarDecl, "customUpdate" + g + "Time", "0.0");
        genHostScalar<Type::Double>(definitionsVar, runnerVarDecl, "customUpdate" + g + "TransposeTime", "0.0");
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
        sg.generateInit(backend, runnerMergedStructAlloc, modelMerged);
    }

    // Generate merged neuron initialisation groups
    for(const auto &m : modelMerged.getMergedNeuronInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through merged synapse init groups
    for(const auto &m : modelMerged.getMergedSynapseInitGroups()) {
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

    // Generate merged custom update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomUpdateInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Generate merged custom WU update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomWUUpdateInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Generate merged custom sparse WU update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomWUUpdateSparseInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Generate merged custom connectivity update presynaptic initialisation groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityUpdatePreInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Generate merged custom connectivity update postsynaptic initialisation groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityUpdatePostInitGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Generate merged custom connectivity update synaptic initialisation groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups()) {
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

    // Loop through custom update host reduction groups
    for(const auto &m : modelMerged.getMergedCustomUpdateHostReductionGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through custom weight update host reduction groups
    for(const auto &m : modelMerged.getMergedCustomWUUpdateHostReductionGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through custom connectivity update groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityUpdateGroups()) {
        m.generateRunner(backend, definitionsInternal, definitionsInternalFunc, definitionsInternalVar,
                         runnerVarDecl, runnerMergedStructAlloc);
    }

    // Loop through custom connectivity host update groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityHostUpdateGroups()) {
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
        backend.genArray<Type::Uint32>(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                       modelMerged.getTypeContext(), "glbSpkCnt" + n.first, 
                                       n.second.getSpikeLocation(), numSpikeCounts, mem);
        backend.genArray<Type::Uint32>(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                      modelMerged.getTypeContext(),  "glbSpk" + n.first, 
                                       n.second.getSpikeLocation(), numSpikes, mem);

        // True spike push and pull functions
        genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeLocation(),
                            backend.getPreferences().automaticCopy, n.first + "Spikes",
                            [&]()
                            {
                                backend.genVariablePushPull<Type::Uint32>(runnerPushFunc, runnerPullFunc,
                                                                          modelMerged.getTypeContext(), "glbSpkCnt" + n.first, 
                                                                          n.second.getSpikeLocation(), true, numSpikeCounts);
                                backend.genVariablePushPull<Type::Uint32>(runnerPushFunc, runnerPullFunc,
                                                                          modelMerged.getTypeContext(), "glbSpk" + n.first, 
                                                                          n.second.getSpikeLocation(), true, numSpikes);
                            });

        // Current true spike getter functions
        genSpikeGetters(definitionsFunc, runnerGetterFunc, n.second, true, batchSize);

        // If spike recording is enabled, define and declare variables and add free
        if(n.second.isSpikeRecordingEnabled()) {
            backend.genVariableDefinition(definitionsVar, definitionsInternalVar, 
                                          Type::Uint32::getInstance(), modelMerged.getTypeContext(), "recordSpk" + n.first, 
                                          VarLocation::HOST_DEVICE);
            backend.genVariableInstantiation(runnerVarDecl, 
                                             Type::Uint32::getInstance(), modelMerged.getTypeContext(), "recordSpk" + n.first, 
                                             VarLocation::HOST_DEVICE);
            backend.genVariableFree(runnerVarFree, 
                                    "recordSpk" + n.first, VarLocation::HOST_DEVICE);
        }

        // If neuron group needs to emit spike-like events
        if (n.second.isSpikeEventRequired()) {
            // Write convenience macros to access spike-like events
            if(batchSize == 1) {
                genSpikeMacros(definitionsVar, n.second, false);
            }

            // Spike-like event variables
            backend.genArray<Type::Uint32>(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                           modelMerged.getTypeContext(), "glbSpkCntEvnt" + n.first, n.second.getSpikeEventLocation(),
                                           batchSize * n.second.getNumDelaySlots(), mem);
            backend.genArray<Type::Uint32>(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                           modelMerged.getTypeContext(), "glbSpkEvnt" + n.first, n.second.getSpikeEventLocation(),
                                           numNeuronDelaySlots, mem);

            // Spike-like event push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeEventLocation(),
                                backend.getPreferences().automaticCopy, n.first + "SpikeEvents",
                                [&]()
                                {
                                    backend.genVariablePushPull<Type::Uint32>(runnerPushFunc, runnerPullFunc,
                                                                              modelMerged.getTypeContext(), "glbSpkCntEvnt" + n.first, 
                                                                              n.second.getSpikeLocation(), true, batchSize * n.second.getNumDelaySlots());
                                    backend.genVariablePushPull<Type::Uint32>(runnerPushFunc, runnerPullFunc, 
                                                                              modelMerged.getTypeContext(), "glbSpkEvnt" + n.first, 
                                                                              n.second.getSpikeLocation(), true, numNeuronDelaySlots);
                                });

            // Current true spike getter functions
            genSpikeGetters(definitionsFunc, runnerGetterFunc, n.second, false, batchSize);

            // If spike recording is enabled, define and declare variables and add free
            if(n.second.isSpikeEventRecordingEnabled()) {
                backend.genVariableDefinition(definitionsVar, definitionsInternalVar, 
                                              Type::Uint32::getInstance(), modelMerged.getTypeContext(), "recordSpkEvent" + n.first, 
                                              VarLocation::HOST_DEVICE);
                backend.genVariableInstantiation(runnerVarDecl, 
                                                 Type::Uint32::getInstance(), modelMerged.getTypeContext(), "recordSpkEvent" + n.first, 
                                                 VarLocation::HOST_DEVICE);
                backend.genVariableFree(runnerVarFree, "recordSpkEvent" + n.first, VarLocation::HOST_DEVICE);
            }
        }

        // If neuron group has axonal delays
        if (n.second.isDelayRequired()) {
            genHostDeviceScalar<Type::Uint32>(modelMerged, backend, definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                              "spkQuePtr" + n.first, "0", mem);
        }

        // If neuron group needs to record its spike times
        if (n.second.isSpikeTimeRequired()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getTimePrecision(), modelMerged.getTypeContext(), "sT" + n.first, 
                             n.second.getSpikeTimeLocation(), numNeuronDelaySlots, mem);

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeTimeLocation(),
                                backend.getPreferences().automaticCopy, n.first + "SpikeTimes",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, 
                                                                model.getTimePrecision(), modelMerged.getTypeContext(), "sT" + n.first, 
                                                                n.second.getSpikeTimeLocation(), true, numNeuronDelaySlots);
                                });
        }

        // If neuron group needs to record its previous spike times
        if (n.second.isPrevSpikeTimeRequired()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getTimePrecision(), modelMerged.getTypeContext(), "prevST" + n.first, 
                             n.second.getPrevSpikeTimeLocation(), numNeuronDelaySlots, mem);

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getPrevSpikeTimeLocation(),
                                backend.getPreferences().automaticCopy, n.first + "PreviousSpikeTimes",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, 
                                                                model.getTimePrecision(), modelMerged.getTypeContext(), "prevST" + n.first, 
                                                                n.second.getPrevSpikeTimeLocation(), true, numNeuronDelaySlots);
                                });
        }

        // If neuron group needs to record its spike-like-event times
        if (n.second.isSpikeEventTimeRequired()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getTimePrecision(), modelMerged.getTypeContext(), "seT" + n.first, 
                             n.second.getSpikeEventTimeLocation(), numNeuronDelaySlots, mem);

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getSpikeTimeLocation(),
                                backend.getPreferences().automaticCopy, n.first + "SpikeEventTimes",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, 
                                                                model.getTimePrecision(), modelMerged.getTypeContext(), "seT" + n.first, 
                                                                n.second.getSpikeEventTimeLocation(), true, numNeuronDelaySlots);
                                });
        }

        // If neuron group needs to record its previous spike-like-event times
        if (n.second.isPrevSpikeEventTimeRequired()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getTimePrecision(), modelMerged.getTypeContext(), "prevSET" + n.first, 
                             n.second.getPrevSpikeEventTimeLocation(), numNeuronDelaySlots, mem);

            // Generate push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getPrevSpikeEventTimeLocation(),
                                backend.getPreferences().automaticCopy, n.first + "PreviousSpikeEventTimes",
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, 
                                                                model.getTimePrecision(), modelMerged.getTypeContext(), "prevSET" + n.first, 
                                                                n.second.getPrevSpikeEventTimeLocation(), true, numNeuronDelaySlots);
                                });
        }

        // If neuron group needs per-neuron RNGs
        if(n.second.isSimRNGRequired()) {
            backend.genPopulationRNG(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     "rng" + n.first, batchSize * n.second.getNumNeurons(), mem);
        }

        // Neuron state variables
        const auto neuronModel = n.second.getNeuronModel();
        std::vector<std::string> neuronStatePushPullFunctions;
        for(const auto &var : neuronModel->getVars()) {
            const auto *varInitSnippet = n.second.getVarInitialisers().at(var.name).getSnippet();
            const unsigned int numCopies = getNumVarCopies(var.access, batchSize);
            const unsigned int numElements = getNumVarElements(var.access, n.second.getNumNeurons());
            const size_t count = n.second.isVarQueueRequired(var.name) ? numCopies * numElements * n.second.getNumDelaySlots() : numCopies * n.second.getNumNeurons();
            const bool autoInitialized = !varInitSnippet->getCode().empty();
            genVariable(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar, 
                        runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerPushFunc, runnerPullFunc, var.type, var.name + n.first,
                        n.second.getVarLocation(var.name), autoInitialized, count, mem, neuronStatePushPullFunctions);

            // Current variable push and pull functions
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, n.second.getVarLocation(var.name),
                                backend.getPreferences().automaticCopy, "Current" + var.name + n.first,
                                [&]()
                                {
                                    backend.genCurrentVariablePushPull(runnerPushFunc, runnerPullFunc, n.second, 
                                                                       var.type, modelMerged.getTypeContext(), var.name, 
                                                                       n.second.getVarLocation(var.name), numCopies);
                                });

            // Write getter to get access to correct pointer
            const bool delayRequired = (n.second.isVarQueueRequired(var.name) &&  n.second.isDelayRequired());
            genVarGetterScope(definitionsFunc, runnerGetterFunc, n.second.getVarLocation(var.name),
                              "Current" + var.name + n.first, var.type->getPointerType()->getResolvedName(modelMerged.getTypeContext()),
                              [&]()
                              {
                                  runnerGetterFunc << "return " << var.name << n.first;
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
            for(const auto &e : varInitSnippet->getExtraGlobalParams()) {
                genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar, 
                                    runnerVarDecl, runnerExtraGlobalParamFunc, 
                                    e.type, e.name + var.name + n.first,
                                    true, VarLocation::HOST_DEVICE);
            }
        }

        // Add helper function to push and pull entire neuron state
        if(!backend.getPreferences().automaticCopy) {
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, 
                             n.first, backend.getPreferences().generateEmptyStatePushPull, 
                             neuronStatePushPullFunctions, statePushPullFunctions);
        }

        for(const auto &e : neuronModel->getExtraGlobalParams()) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc, 
                                e.type, e.name + n.first,
                                true, n.second.getExtraGlobalParamLocation(e.name));
        }

        if(!n.second.getCurrentSources().empty()) {
            allVarStreams << "// current source variables" << std::endl;
        }
        for (auto const *cs : n.second.getCurrentSources()) {
            std::vector<std::string> currentSourcePushPullFunctions;
            genRunnerVars<CurrentSourceVarAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                   runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
                                                   runnerPushFunc, runnerPullFunc, *cs, mem, currentSourcePushPullFunctions,
                                                   [batchSize, &n](const CurrentSourceInternal&, const Models::Base::Var &var)
                                                   { 
                                                       return getVarSize(var.access, n.second.getNumNeurons(), batchSize);
                                                   });
            genRunnerEGPs<CurrentSourceEGPAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                   runnerVarDecl, runnerExtraGlobalParamFunc, *cs);
            // Add helper function to push and pull entire group state
            if(!backend.getPreferences().automaticCopy) {
                genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc,
                                 cs->getName(), backend.getPreferences().generateEmptyStatePushPull,
                                 currentSourcePushPullFunctions, statePushPullFunctions);
            }
        }
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// custom update variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    genCustomUpdate<CustomUpdateVarAdapter, CustomUpdateEGPAdapter>(
        modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
        runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
        runnerPushFunc, runnerPullFunc, model.getCustomUpdates(), mem, statePushPullFunctions,
        [batchSize](const CustomUpdateInternal &c, const Models::Base::Var &var)
        { 
            return getVarSize(var.access, c.getSize(), batchSize, 1, c.isBatched());
        });

    genCustomUpdate<CustomUpdateVarAdapter, CustomUpdateEGPAdapter>(
        modelMerged, backend,
        definitionsVar, definitionsFunc, definitionsInternalVar,
        runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
        runnerPushFunc, runnerPullFunc, model.getCustomWUUpdates(), mem, statePushPullFunctions, 
        [batchSize, &backend](const CustomUpdateWUInternal &c, const Models::Base::Var &var) 
        { 
            const SynapseGroupInternal *sg = c.getSynapseGroup();
            const size_t count = ((sg->getMatrixType() & SynapseMatrixWeight::KERNEL)
                                  ? sg->getKernelSizeFlattened() 
                                  : sg->getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(*sg));
            return getVarSize(var.access, count, batchSize, 1, c.isBatched());
        });
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// custom connectivity update variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    // Loop through custom updates
    for(const auto &c : model.getCustomConnectivityUpdates()) {
        // Generate variables
        std::vector<std::string> customConnectivityPushPullFunctions;
        genRunnerVars<CustomConnectivityUpdateVarAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                          runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
                                                          runnerPushFunc, runnerPullFunc, c.second, mem, customConnectivityPushPullFunctions,
                                                          [&backend](const CustomConnectivityUpdateInternal &c, const Models::Base::Var&)
                                                          { 
                                                              const SynapseGroupInternal *sg = c.getSynapseGroup();
                                                              return (sg->getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(*sg));
                                                          });
        
        // Generate presynaptic variables
        genRunnerVars<CustomConnectivityUpdatePreVarAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                             runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
                                                             runnerPushFunc, runnerPullFunc, c.second, mem, customConnectivityPushPullFunctions,
                                                             [](const CustomConnectivityUpdateInternal &c, const Models::Base::Var&)
                                                             { 
                                                                 return c.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons();
                                                             });


        // Generate postsynaptic variables
        genRunnerVars<CustomConnectivityUpdatePostVarAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                              runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc,
                                                              runnerPushFunc, runnerPullFunc, c.second, mem, customConnectivityPushPullFunctions,
                                                              [&backend](const CustomConnectivityUpdateInternal &c, const Models::Base::Var&)
                                                              { 
                                                                  return c.getSynapseGroup()->getTrgNeuronGroup()->getNumNeurons();
                                                              });

        // Generate EGPs
        genRunnerEGPs<CustomConnectivityUpdateEGPAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                          runnerVarDecl, runnerExtraGlobalParamFunc, c.second);

        // If custom connectivity update group needs per-row RNGs
        if(c.second.isRowSimRNGRequired()) {
            backend.genPopulationRNG(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                     "rowRNG" + c.first, c.second.getSynapseGroup()->getSrcNeuronGroup()->getNumNeurons(), mem);
        }

        
        // Add helper function to push and pull entire group state
        if(!backend.getPreferences().automaticCopy) {
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc,
                             c.first, backend.getPreferences().generateEmptyStatePushPull,
                             customConnectivityPushPullFunctions, statePushPullFunctions);
        }
    }


    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// pre and postsynaptic variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    for(const auto &n : model.getNeuronGroups()) {
        // Loop through merged postsynaptic models of incoming synaptic populations
        for(const auto *sg : n.second.getFusedPSMInSyn()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getPrecision(), modelMerged.getTypeContext(), "inSyn" + sg->getFusedPSVarSuffix(), 
                             sg->getInSynLocation(), sg->getTrgNeuronGroup()->getNumNeurons() * batchSize, mem);

            if (sg->isDendriticDelayRequired()) {
                backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 model.getPrecision(), modelMerged.getTypeContext(), "denDelay" + sg->getFusedPSVarSuffix(), 
                                 sg->getDendriticDelayLocation(), (size_t)sg->getMaxDendriticDelayTimesteps() * (size_t)sg->getTrgNeuronGroup()->getNumNeurons() * batchSize, mem);
                genHostDeviceScalar<Type::Uint32>(modelMerged, backend, definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                                  "denDelayPtr" + sg->getFusedPSVarSuffix(), "0", mem);
            }

            genRunnerFusedVars<SynapsePSMVarAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                     runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc, *sg, mem, 
                                                     [batchSize](const SynapseGroupInternal &sg, const Models::Base::Var &var)
                                                     { 
                                                         return getVarSize(var.access, sg.getTrgNeuronGroup()->getNumNeurons(), batchSize);
                                                     }); 
        }
        // Loop through fused outgoing synapse populations with weightupdate models that have presynaptic output 
        for(const auto *sg : n.second.getFusedPreOutputOutSyn()) {
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             model.getPrecision(), modelMerged.getTypeContext(), "revInSyn" + sg->getFusedPreOutputSuffix(), 
                             sg->getInSynLocation(), sg->getSrcNeuronGroup()->getNumNeurons() * batchSize, mem);
        }
        
        // Loop through merged postsynaptic weight updates of incoming synaptic populations
        for(const auto *sg: n.second.getFusedWUPreOutSyn()) {
            const unsigned int preDelaySlots = (sg->getDelaySteps() == NO_DELAY) ? 1 : sg->getSrcNeuronGroup()->getNumDelaySlots();
            genRunnerFusedVars<SynapseWUPreVarAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                       runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc, *sg, mem, 
                                                       [batchSize, preDelaySlots](const SynapseGroupInternal &sg, const Models::Base::Var &var)
                                                       { 
                                                           return getVarSize(var.access, sg.getSrcNeuronGroup()->getNumNeurons(), 
                                                                             batchSize, preDelaySlots);
                                                       }); 
        }
        
        // Loop through merged postsynaptic weight updates of incoming synaptic populations
        for(const auto *sg: n.second.getFusedWUPostInSyn()) { 
            const unsigned int postDelaySlots = (sg->getBackPropDelaySteps() == NO_DELAY) ? 1 : sg->getTrgNeuronGroup()->getNumDelaySlots();
            genRunnerFusedVars<SynapseWUPostVarAdapter>(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                                        runnerVarDecl, runnerVarAlloc, runnerVarFree, runnerExtraGlobalParamFunc, *sg, mem, 
                                                        [batchSize, postDelaySlots](const SynapseGroupInternal &sg, const Models::Base::Var &var)
                                                        { 
                                                            return getVarSize(var.access, sg.getTrgNeuronGroup()->getNumNeurons(), 
                                                                              batchSize, postDelaySlots);
                                                        }); 
        }  
    }
    allVarStreams << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// synapse connectivity" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    std::vector<std::string> connectivityPushPullFunctions;
    for(const auto &s : model.getSynapseGroups()) {
        const auto *snippet = s.second.getConnectivityInitialiser().getSnippet();
        const bool autoInitialized = !snippet->getRowBuildCode().empty() || !snippet->getColBuildCode().empty();

        if(s.second.getMatrixType() & SynapseMatrixConnectivity::BITMASK) {
            const size_t gpSize = ceilDivide((size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * backend.getSynapticMatrixRowStride(s.second), 32);
            backend.genArray<Type::Uint32>(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                           modelMerged.getTypeContext(), "gp" + s.second.getName(), 
                                           s.second.getSparseConnectivityLocation(), gpSize, mem);

            // Generate push and pull functions for bitmask
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getSparseConnectivityLocation(),
                                backend.getPreferences().automaticCopy, s.second.getName() + "Connectivity", connectivityPushPullFunctions,
                                [&]()
                                {
                                    // Row lengths
                                    backend.genVariablePushPull<Type::Uint32>(runnerPushFunc, runnerPullFunc, 
                                                                              modelMerged.getTypeContext(), "gp" + s.second.getName(),
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
            backend.genArray<Type::Uint32>(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                           modelMerged.getTypeContext(), "rowLength" + s.second.getName(), 
                                           varLoc, s.second.getSrcNeuronGroup()->getNumNeurons(), mem);

            // Target indices
            backend.genArray(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             s.second.getSparseIndType(),  modelMerged.getTypeContext(), "ind" + s.second.getName(), 
                             varLoc, size, mem);

            // **TODO** remap is not always required
            if(backend.isPostsynapticRemapRequired() && !s.second.getWUModel()->getLearnPostCode().empty()) {
                const size_t postSize = (size_t)s.second.getTrgNeuronGroup()->getNumNeurons() * (size_t)s.second.getMaxSourceConnections();

                // Allocate column lengths
                backend.genArray<Type::Uint32>(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                               modelMerged.getTypeContext(), "colLength" + s.second.getName(), 
                                               VarLocation::DEVICE, s.second.getTrgNeuronGroup()->getNumNeurons(), mem);

                // Allocate remap
                backend.genArray<Type::Uint32>(definitionsVar, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                               modelMerged.getTypeContext(), "remap" + s.second.getName(), 
                                               VarLocation::DEVICE, postSize, mem);
            }

            // Generate push and pull functions for sparse connectivity
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getSparseConnectivityLocation(),
                                backend.getPreferences().automaticCopy, s.second.getName() + "Connectivity", connectivityPushPullFunctions,
                                [&]()
                                {
                                    // Row lengths
                                    backend.genVariablePushPull<Type::Uint32>(runnerPushFunc, runnerPullFunc, 
                                                                              modelMerged.getTypeContext(), "rowLength" + s.second.getName(), 
                                                                              s.second.getSparseConnectivityLocation(), autoInitialized, s.second.getSrcNeuronGroup()->getNumNeurons());

                                    // Target indices
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc,  
                                                                s.second.getSparseIndType(), modelMerged.getTypeContext(), "ind" + s.second.getName(), 
                                                                s.second.getSparseConnectivityLocation(), autoInitialized, size);
                                });
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
        const bool kernelWeights = (s.second.getMatrixType() & SynapseMatrixWeight::KERNEL);
        const bool proceduralWeights = (s.second.getMatrixType() & SynapseMatrixWeight::PROCEDURAL);
        std::vector<std::string> synapseGroupStatePushPullFunctions;
        if (individualWeights || proceduralWeights || kernelWeights) {
            for(const auto &wuVar : wu->getVars()) {
                const auto *varInitSnippet = s.second.getWUVarInitialisers().at(wuVar.name).getSnippet();
                const bool autoInitialized = !varInitSnippet->getCode().empty();
                if(individualWeights) {
                    const size_t size = (size_t)s.second.getSrcNeuronGroup()->getNumNeurons() * (size_t)backend.getSynapticMatrixRowStride(s.second);
                    genVariable(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                runnerPushFunc, runnerPullFunc, wuVar.type, wuVar.name + s.second.getName(), s.second.getWUVarLocation(wuVar.name),
                                autoInitialized, size * getNumVarCopies(wuVar.access, batchSize), mem, synapseGroupStatePushPullFunctions);
                }
                else if(kernelWeights) {
                     // Calculate size of kernel
                     const size_t size = s.second.getKernelSizeFlattened() * getNumVarCopies(wuVar.access, batchSize);
                     
                     // Generate variable
                     genVariable(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                                 runnerPushFunc, runnerPullFunc, wuVar.type, wuVar.name + s.second.getName(), s.second.getWUVarLocation(wuVar.name),
                                 autoInitialized, size, mem, synapseGroupStatePushPullFunctions);
                }

                // Loop through EGPs required to initialize WUM 
                for(const auto &e : varInitSnippet->getExtraGlobalParams()) {
                    genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                        runnerVarDecl, runnerExtraGlobalParamFunc, 
                                        e.type, e.name + wuVar.name + s.second.getName(),
                                        true, VarLocation::HOST_DEVICE);
                }
            }
        }

        // If this synapse group's postsynaptic models hasn't been merged (which makes pulling them somewhat ambiguous)
        // **NOTE** we generated initialisation and declaration code earlier - here we just generate push and pull as we want this per-synapse group
        if(!s.second.isPSModelFused()) {
            // Add code to push and pull inSyn
            genVarPushPullScope(definitionsFunc, runnerPushFunc, runnerPullFunc, s.second.getInSynLocation(),
                                backend.getPreferences().automaticCopy, "inSyn" + s.second.getName(), synapseGroupStatePushPullFunctions,
                                [&]()
                                {
                                    backend.genVariablePushPull(runnerPushFunc, runnerPullFunc, 
                                                                model.getPrecision(), modelMerged.getTypeContext(), "inSyn" + s.second.getName(), 
                                                                s.second.getInSynLocation(), true, s.second.getTrgNeuronGroup()->getNumNeurons() * batchSize);
                                });

            genRunnerFusedVarPushPull<SynapsePSMVarAdapter>(modelMerged, backend, definitionsFunc, runnerPushFunc, runnerPullFunc, s.second, synapseGroupStatePushPullFunctions,
                                                            [batchSize](const SynapseGroupInternal &sg, const Models::Base::Var &var)
                                                            { 
                                                                return getVarSize(var.access, sg.getTrgNeuronGroup()->getNumNeurons(), batchSize);
                                                            }); 
        }
        
        // If this synapse group's presynaptic weight updates hasn't been merged (which makes pulling them somewhat ambiguous)
        // **NOTE** we generated initialisation and declaration code earlier - here we just generate push and pull as we want this per-synapse group
        if(!s.second.isWUPreModelFused()) {
            const unsigned int preDelaySlots = (s.second.getDelaySteps() == NO_DELAY) ? 1 : s.second.getSrcNeuronGroup()->getNumDelaySlots();
            genRunnerFusedVarPushPull<SynapseWUPreVarAdapter>(modelMerged, backend, definitionsFunc, runnerPushFunc, runnerPullFunc, s.second, synapseGroupStatePushPullFunctions,
                                                              [batchSize, preDelaySlots](const SynapseGroupInternal &sg, const Models::Base::Var &var)
                                                              { 
                                                                  return getVarSize(var.access, sg.getSrcNeuronGroup()->getNumNeurons(), batchSize, preDelaySlots);
                                                              }); 
            
        }
        
        // If this synapse group's postsynaptic weight updates hasn't been merged (which makes pulling them somewhat ambiguous)
        // **NOTE** we generated initialisation and declaration code earlier - here we just generate push and pull as we want this per-synapse group
        if(!s.second.isWUPostModelFused()) {
            const unsigned int postDelaySlots = (s.second.getBackPropDelaySteps() == NO_DELAY) ? 1 : s.second.getTrgNeuronGroup()->getNumDelaySlots();
            genRunnerFusedVarPushPull<SynapseWUPostVarAdapter>(modelMerged, backend, definitionsFunc, runnerPushFunc, runnerPullFunc, s.second, synapseGroupStatePushPullFunctions,
                                                              [batchSize, postDelaySlots](const SynapseGroupInternal &sg, const Models::Base::Var &var)
                                                              { 
                                                                  return getVarSize(var.access, sg.getTrgNeuronGroup()->getNumNeurons(), batchSize, postDelaySlots);
                                                              });
        }
        
        // Add helper function to push and pull entire synapse group state
        if(!backend.getPreferences().automaticCopy) {
            genStatePushPull(definitionsFunc, runnerPushFunc, runnerPullFunc, 
                             s.second.getName(), backend.getPreferences().generateEmptyStatePushPull, 
                             synapseGroupStatePushPullFunctions, statePushPullFunctions);
        }

        // **NOTE** postsynaptic models aren't allowed in merged groups so it's fine to do this here
        for(const auto &e : psm->getExtraGlobalParams()) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc, 
                                e.type, e.name + s.second.getName(),
                                true, s.second.getPSExtraGlobalParamLocation(e.name));
        }

        for(const auto &e : wu->getExtraGlobalParams()) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc, 
                                e.type, e.name + s.second.getName(),
                                true, s.second.getWUExtraGlobalParamLocation(e.name));
        }

        const auto sparseConnExtraGlobalParams = s.second.getConnectivityInitialiser().getSnippet()->getExtraGlobalParams();
        for(const auto &e : sparseConnExtraGlobalParams) {
            genExtraGlobalParam(modelMerged, backend, definitionsVar, definitionsFunc, definitionsInternalVar,
                                runnerVarDecl, runnerExtraGlobalParamFunc, 
                                e.type, e.name + s.second.getName(),
                                s.second.getConnectivityInitialiser().getSnippet()->getHostInitCode().empty(),
                                s.second.getSparseConnectivityExtraGlobalParamLocation(e.name));
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
                if(n.second.isSpikeRecordingEnabled()) {
                    CodeStream::Scope b(runner);
                    backend.genVariableDynamicAllocation(runner, 
                                                         Type::Uint32::getInstance(), modelMerged.getTypeContext(), "recordSpk" + n.first, 
                                                         VarLocation::HOST_DEVICE, "numWords");

                    // Get destinations in merged structures, this EGP 
                    // needs to be copied to and call push function
                    const auto &mergedDestinations = modelMerged.getMergedEGPDestinations(backend.getDeviceVarPrefix() + "recordSpk" + n.first);
                    for(const auto &v : mergedDestinations) {
                        runner << "pushMerged" << v.first << v.second.mergedGroupIndex << v.second.fieldName << "ToDevice(";
                        runner << v.second.groupIndex << ", " << backend.getDeviceVarPrefix() << "recordSpk" + n.first << ");" << std::endl;
                    }
                }

                // Allocate spike event array if required
                if(n.second.isSpikeEventRecordingEnabled()) {
                    CodeStream::Scope b(runner);
                    backend.genVariableDynamicAllocation(runner, 
                                                         Type::Uint32::getInstance(), modelMerged.getTypeContext(), "recordSpkEvent" + n.first, 
                                                         VarLocation::HOST_DEVICE, "numWords");

                    // Get destinations in merged structures, this EGP 
                    // needs to be copied to and call push function
                    const auto &mergedDestinations = modelMerged.getMergedEGPDestinations(backend.getDeviceVarPrefix() + "recordSpkEvent" + n.first);
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
                if(n.second.isSpikeRecordingEnabled()) {
                    CodeStream::Scope b(runner);
                    backend.genVariableDynamicPull(runner, 
                                                   Type::Uint32::getInstance(), modelMerged.getTypeContext(), "recordSpk" + n.first, 
                                                   VarLocation::HOST_DEVICE, "numWords");
                }
                // AllocaPullte spike event array if required
                // **YUCK** maybe this should be renamed pullDynamicArray
                if(n.second.isSpikeEventRecordingEnabled()) {
                    CodeStream::Scope b(runner);
                    backend.genVariableDynamicPull(runner, 
                                                   Type::Uint32::getInstance(), modelMerged.getTypeContext(), "recordSpkEvent" + n.first, 
                                                   VarLocation::HOST_DEVICE, "numWords");
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

        // Generate preamble - this is the first bit of generated code called by user simulations
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

        // Generate backend-specific preamble
        backend.genFreeMemPreamble(runner, modelMerged);

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
            for(const auto *sg : n.second.getFusedPSMInSyn()) {
                if(sg->isDendriticDelayRequired()) {
                    runner << "denDelayPtr" << sg->getFusedPSVarSuffix() << " = (denDelayPtr" << sg->getFusedPSVarSuffix() << " + 1) % " << sg->getMaxDendriticDelayTimesteps() << ";" << std::endl;
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
    definitions << "EXPORT_FUNC void updateNeurons(" << model.getTimePrecision()->getName() << " t";
    if(model.isRecordingInUse()) {
        definitions << ", unsigned int recordingTimestep";
    }
    definitions << "); " << std::endl;
    definitions << "EXPORT_FUNC void updateSynapses(" << model.getTimePrecision()->getName() << " t);" << std::endl;
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
