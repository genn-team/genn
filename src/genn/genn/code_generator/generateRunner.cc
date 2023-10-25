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
void genHostScalar(CodeStream &definitionsVar, CodeStream &runnerVarDecl,
                   const Type::ResolvedType &type, const std::string &name, const std::string &value)
{
    definitionsVar << "EXPORT_VAR " << type.getValue().name << " " << name << ";" << std::endl;
    runnerVarDecl << type.getValue().name << " " << name << " = " << value << ";" << std::endl;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// GeNN::CodeGenerator
//--------------------------------------------------------------------------
void GeNN::CodeGenerator::generateRunner(const filesystem::path &outputPath, ModelSpecMerged &modelMerged, 
                                             const BackendBase &backend, const std::string &suffix)
{
    // Create output streams to write to file and wrap in CodeStreams
    std::ofstream definitionsStream((outputPath / ("definitions" + suffix + ".h")).str());
    std::ofstream runnerStream((outputPath / ("runner" + suffix + ".cc")).str());
    CodeStream definitions(definitionsStream);
    CodeStream runner(runnerStream);

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
    
    // Write ranges of scalar and time types
    const ModelSpecInternal &model = modelMerged.getModel();
    genTypeRange(definitions, model.getPrecision(), "SCALAR");
    genTypeRange(definitions, model.getTimePrecision(), "TIME");

    // Write runner preamble
    runner << "#include \"definitions" << suffix << ".h\"" << std::endl << std::endl;

    // Create codestreams to generate different sections of runner and definitions
	std::stringstream runnerVarDeclStream;
	std::stringstream runnerVarAllocStream;
	std::stringstream runnerVarFreeStream;
    std::stringstream runnerStepTimeFinaliseStream;
    std::stringstream definitionsVarStream;
    std::stringstream definitionsFuncStream;
    CodeStream runnerVarDecl(runnerVarDeclStream);
    CodeStream runnerVarAlloc(runnerVarAllocStream);
    CodeStream runnerVarFree(runnerVarFreeStream);
    CodeStream runnerStepTimeFinalise(runnerStepTimeFinaliseStream);
    CodeStream definitionsVar(definitionsVarStream);
    CodeStream definitionsFunc(definitionsFuncStream);
    
    // Create a teestream to allow simultaneous writing to all streams
    TeeStream allVarStreams(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree);

    // Begin extern C block around variable declarations
    runnerVarDecl << "extern \"C\" {" << std::endl;
    definitionsVar << "extern \"C\" {" << std::endl;

    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;
    allVarStreams << "// global variables" << std::endl;
    allVarStreams << "// ------------------------------------------------------------------------" << std::endl;

    // If backend requires a global device RNG to simulate (or initialize) this model
    if(backend.isGlobalDeviceRNGRequired(model)) {
        backend.genGlobalDeviceRNG(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree);
    }
    
    // If backend required a global host RNG to simulate (or initialize) this model, generate a standard Mersenne Twister
    if(backend.isGlobalHostRNGRequired(model)) {
        // Define standard RNG
        definitionsVar << "extern " << "std::mt19937 hostRNG;" << std::endl;
        runnerVarDecl << "std::mt19937 hostRNG;" << std::endl;

        // Define standard host distributions as recreating them each call is slow
        definitionsVar << "extern " << "std::uniform_real_distribution<" << model.getPrecision().getName() << "> standardUniformDistribution;" << std::endl;
        definitionsVar << "extern " << "std::normal_distribution<" << model.getPrecision().getName() << "> standardNormalDistribution;" << std::endl;
        definitionsVar << "extern " << "std::exponential_distribution<" << model.getPrecision().getName() << "> standardExponentialDistribution;" << std::endl;
        definitionsVar << std::endl;
        runnerVarDecl << "std::uniform_real_distribution<" << model.getPrecision().getName() << "> standardUniformDistribution(" << Type::writeNumeric(0.0, model.getPrecision()) << ", " << Type::writeNumeric(1.0, model.getPrecision()) << ");" << std::endl;
        runnerVarDecl << "std::normal_distribution<" << model.getPrecision().getName() << "> standardNormalDistribution(" << Type::writeNumeric(0.0, model.getPrecision()) << ", " << Type::writeNumeric(1.0, model.getPrecision()) << ");" << std::endl;
        runnerVarDecl << "std::exponential_distribution<" << model.getPrecision().getName() << "> standardExponentialDistribution(" << Type::writeNumeric(1.0, model.getPrecision()) << ");" << std::endl;
        runnerVarDecl << std::endl;

        // If no seed is specified, use system randomness to generate seed sequence
        CodeStream::Scope b(runnerVarAlloc);
        if(model.getSeed() == 0) {
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
            runnerVarAlloc << "std::seed_seq seeds{" << model.getSeed() << "};" << std::endl;
        }

        // Seed RNG from seed sequence
        runnerVarAlloc << "hostRNG.seed(seeds);" << std::endl;
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
    genHostScalar(definitionsVar, runnerVarDecl, Type::Double, "initTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, Type::Double, "initSparseTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, Type::Double, "neuronUpdateTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, Type::Double, "presynapticUpdateTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, Type::Double, "postsynapticUpdateTime", "0.0");
    genHostScalar(definitionsVar, runnerVarDecl, Type::Double, "synapseDynamicsTime", "0.0");

    // Generate variables to store total elapsed time for each custom update group
    for(const auto &g : customUpdateGroups) {
        genHostScalar(definitionsVar, runnerVarDecl, Type::Double, "customUpdate" + g + "Time", "0.0");
        genHostScalar(definitionsVar, runnerVarDecl, Type::Double, "customUpdate" + g + "TransposeTime", "0.0");
    }
    
    // If timing is actually enabled
    if(model.isTimingEnabled()) {
        // Create neuron timer
        backend.genTimer(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         runnerStepTimeFinalise, "neuronUpdate", true);

        // Add presynaptic update timer
        if(!modelMerged.getMergedPresynapticUpdateGroups().empty()) {
            backend.genTimer(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "presynapticUpdate", true);
        }

        // Add postsynaptic update timer if required
        if(!modelMerged.getMergedPostsynapticUpdateGroups().empty()) {
            backend.genTimer(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "postsynapticUpdate", true);
        }

        // Add synapse dynamics update timer if required
        if(!modelMerged.getMergedSynapseDynamicsGroups().empty()) {
            backend.genTimer(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "synapseDynamics", true);
        }

        // Add timers for each custom update group
        for(const auto &g : customUpdateGroups) {
            backend.genTimer(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "customUpdate" + g, false);
            backend.genTimer(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "customUpdate" + g + "Transpose", false);
        }

        // Create init timer
        backend.genTimer(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                         runnerStepTimeFinalise, "init", false);

        // Add sparse initialisation timer
        if(!modelMerged.getMergedSynapseSparseInitGroups().empty()) {
            backend.genTimer(definitionsVar, runnerVarDecl, runnerVarAlloc, runnerVarFree,
                             runnerStepTimeFinalise, "initSparse", false);
        }

        allVarStreams << std::endl;
    }

    runnerVarDecl << "// ------------------------------------------------------------------------" << std::endl;
    runnerVarDecl << "// merged group arrays" << std::endl;
    runnerVarDecl << "// ------------------------------------------------------------------------" << std::endl;

    // End extern C block around variable declarations
    runnerVarDecl << "}  // extern \"C\"" << std::endl;
 
    // Write pre-amble to runner
    backend.genRunnerPreamble(runner, modelMerged);

    // Write variable declarations to runner
    runner << runnerVarDeclStream.str();

   
    // ---------------------------------------------------------------------
    // Function for setting the device and the host's global variables.
    // Also estimates memory usage on device ...
    runner << "void allocateMem()";
    {
        CodeStream::Scope b(runner);

        // Generate preamble - this is the first bit of generated code called by user simulations
        // so global initialisation is often performed here
        backend.genAllocateMemPreamble(runner, modelMerged);

        // Write variable allocations to runner
        runner << runnerVarAllocStream.str();
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
    // Function to free all global memory structures
    runner << "void stepTime(unsigned long long timestep, unsigned long long numRecordingTimesteps)";
    {
        CodeStream::Scope b(runner);

        runner << "const " << model.getTimePrecision().getName() << " t = timestep * " << Type::writeNumeric(model.getDT(), model.getTimePrecision()) << ";" << std::endl;

        // Update synaptic state
        runner << "updateSynapses(t);" << std::endl;

        // Update neuronal state
        runner << "updateNeurons(t";
        if(model.isRecordingInUse()) {
            runner << ", (unsigned int)(timestep % numRecordingTimesteps)";
        }
        runner << "); " << std::endl;

        // Write step time finalise logic to runner
        runner << runnerStepTimeFinaliseStream.str();
    }
    runner << std::endl;

    // Write variable and function definitions to header
    definitions << definitionsVarStream.str();
    definitions << definitionsFuncStream.str();

    // ---------------------------------------------------------------------
    // Function definitions
    definitions << "// Runner functions" << std::endl;
    definitions << "EXPORT_FUNC void allocateMem();" << std::endl;
    definitions << "EXPORT_FUNC void freeMem();" << std::endl;
    definitions << "EXPORT_FUNC void stepTime(unsigned long long timestep, unsigned long long numRecordingTimesteps);" << std::endl;
    definitions << std::endl;
    definitions << "// Functions generated by backend" << std::endl;
    definitions << "EXPORT_FUNC void updateNeurons(" << modelMerged.getModel().getTimePrecision().getName() << " t";
    if(model.isRecordingInUse()) {
        definitions << ", unsigned int recordingTimestep";
    }
    definitions << "); " << std::endl;
    definitions << "EXPORT_FUNC void updateSynapses(" << modelMerged.getModel().getTimePrecision().getName() << " t);" << std::endl;
    definitions << "EXPORT_FUNC void initialize();" << std::endl;
    definitions << "EXPORT_FUNC void initializeSparse();" << std::endl;
    definitions << "EXPORT_FUNC void initializeHost();" << std::endl;
    
    // Generate function definitions for each custom update
    for(const auto &g : customUpdateGroups) {
        definitions << "EXPORT_FUNC void update" << g << "();" << std::endl;
    }
    definitions << std::endl;

    // Loop through merged synapse connectivity host initialisation groups
    definitions << "// Merged group upload functions" << std::endl;
    for(const auto &m : modelMerged.getMergedSynapseConnectivityHostInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Generate merged neuron initialisation groups
    for(const auto &m : modelMerged.getMergedNeuronInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through merged synapse init groups
    for(const auto &m : modelMerged.getMergedSynapseInitGroups()) {
         m.generateRunner(backend, definitions);
    }

    // Loop through merged synapse connectivity initialisation groups
    for(const auto &m : modelMerged.getMergedSynapseConnectivityInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through merged sparse synapse init groups
    for(const auto &m : modelMerged.getMergedSynapseSparseInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Generate merged custom update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomUpdateInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Generate merged custom WU update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomWUUpdateInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Generate merged custom sparse WU update initialisation groups
    for(const auto &m : modelMerged.getMergedCustomWUUpdateSparseInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Generate merged custom connectivity update presynaptic initialisation groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityUpdatePreInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Generate merged custom connectivity update postsynaptic initialisation groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityUpdatePostInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Generate merged custom connectivity update synaptic initialisation groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityUpdateSparseInitGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through merged neuron update groups
    for(const auto &m : modelMerged.getMergedNeuronUpdateGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through merged presynaptic update groups
    for(const auto &m : modelMerged.getMergedPresynapticUpdateGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through merged postsynaptic update groups
    for(const auto &m : modelMerged.getMergedPostsynapticUpdateGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through synapse dynamics groups
    for(const auto &m : modelMerged.getMergedSynapseDynamicsGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through neuron groups whose previous spike times need resetting
    for(const auto &m : modelMerged.getMergedNeuronPrevSpikeTimeUpdateGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through neuron groups whose spike queues need resetting
    for(const auto &m : modelMerged.getMergedNeuronSpikeQueueUpdateGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through synapse groups whose dendritic delay pointers need updating
    for(const auto &m : modelMerged.getMergedSynapseDendriticDelayUpdateGroups()) {
       m.generateRunner(backend, definitions);
    }
    
    // Loop through custom variable update groups
    for(const auto &m : modelMerged.getMergedCustomUpdateGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through custom WU variable update groups
    for(const auto &m : modelMerged.getMergedCustomUpdateWUGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through custom WU transpose variable update groups
    for(const auto &m : modelMerged.getMergedCustomUpdateTransposeWUGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through custom update host reduction groups
    for(const auto &m : modelMerged.getMergedCustomUpdateHostReductionGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through custom weight update host reduction groups
    for(const auto &m : modelMerged.getMergedCustomWUUpdateHostReductionGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through custom connectivity update groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityUpdateGroups()) {
        m.generateRunner(backend, definitions);
    }

    // Loop through custom connectivity host update groups
    for(const auto &m : modelMerged.getMergedCustomConnectivityHostUpdateGroups()) {
        m.generateRunner(backend, definitions);
    }

    // End extern C block around definitions
    definitions << "}  // extern \"C\"" << std::endl;
}
