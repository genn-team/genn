#include "code_generator/generateAll.h"

// Standard C++ includes
#include <fstream>
#include <string>
#include <vector>

// PLOG includes
#include <plog/Log.h>

// Filesystem includes
#include "path.h"

// GeNN includes
#include "modelSpecInternal.h"

// Code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/generateCustomUpdate.h"
#include "code_generator/generateInit.h"
#include "code_generator/generateNeuronUpdate.h"
#include "code_generator/generateSupportCode.h"
#include "code_generator/generateSynapseUpdate.h"
#include "code_generator/generateRunner.h"
#include "code_generator/modelSpecMerged.h"

//--------------------------------------------------------------------------
// Anonymous namespace
//--------------------------------------------------------------------------
namespace
{
void copyFile(const filesystem::path &file, const filesystem::path &sharePath, const filesystem::path &outputPath)
{
    // Get full path to input and output files
    const auto inputFile = sharePath / file;
    const auto outputFile = outputPath / file;

    // Assert that input file exists
    assert(inputFile.exists());

    // Create output directory if required
    filesystem::create_directory_recursive(outputFile.parent_path());

    // Copy file
    // **THINK** we could check modification etc but it doesn't seem worthwhile
    LOGD_CODE_GEN << "Copying '" << inputFile << "' to '" << outputFile << "'" << std::endl;
    std::ifstream inputFileStream(inputFile.str(), std::ios::binary);
    std::ofstream outputFileStream(outputFile.str(), std::ios::binary);
    assert(outputFileStream.good());
    outputFileStream << inputFileStream.rdbuf();
}
}

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
std::pair<std::vector<std::string>, CodeGenerator::MemAlloc> CodeGenerator::generateAll(const ModelSpecInternal &model, const BackendBase &backend,
                                                                                        const filesystem::path &sharePath, const filesystem::path &outputPath,
                                                                                        bool standaloneModules)
{
    // Create directory for generated code
    filesystem::create_directory(outputPath);

    // Open output file streams for generated code files
    std::ofstream definitionsStream((outputPath / "definitions.h").str());
    std::ofstream definitionsInternalStream((outputPath / "definitionsInternal.h").str());
    std::ofstream customUpdateStream((outputPath / "customUpdate.cc").str());
    std::ofstream neuronUpdateStream((outputPath / "neuronUpdate.cc").str());
    std::ofstream synapseUpdateStream((outputPath / "synapseUpdate.cc").str());
    std::ofstream initStream((outputPath / "init.cc").str());
    std::ofstream runnerStream((outputPath / "runner.cc").str());

    // Wrap output file streams in CodeStreams for formatting
    CodeStream definitions(definitionsStream);
    CodeStream definitionsInternal(definitionsInternalStream);
    CodeStream customUpdate(customUpdateStream);
    CodeStream neuronUpdate(neuronUpdateStream);
    CodeStream synapseUpdate(synapseUpdateStream);
    CodeStream init(initStream);
    CodeStream runner(runnerStream);

    // Create merged model
    ModelSpecMerged modelMerged(model, backend);

    // Generate modules
    //**NOTE** memory spaces are given out on a first-come, first-serve basis so the modules should be in preferential order
    auto memorySpaces = backend.getMergedGroupMemorySpaces(modelMerged);
    auto mem = generateRunner(definitions, definitionsInternal, runner, modelMerged, backend);
    generateSynapseUpdate(synapseUpdate, memorySpaces, modelMerged, backend);
    generateNeuronUpdate(neuronUpdate, memorySpaces, modelMerged, backend);
    generateCustomUpdate(customUpdate, memorySpaces, modelMerged, backend);
    generateInit(init, memorySpaces, modelMerged, backend);

    // Generate support code module if the backend supports namespaces
    if (backend.supportsNamespace()) {
        std::ofstream supportCodeStream((outputPath / "supportCode.h").str());
        CodeStream supportCode(supportCodeStream);
        generateSupportCode(supportCode, modelMerged);
    }

    // Get list of files to copy into generated code
    const auto backendSharePath = sharePath / "backends";
    const auto filesToCopy = backend.getFilesToCopy(modelMerged);
    const auto absOutputPath = outputPath.make_absolute();
    for(const auto &f : filesToCopy) {
        copyFile(f, backendSharePath, absOutputPath);
    }

    // Create basic list of modules
    std::vector<std::string> modules = {"customUpdate", "neuronUpdate", "synapseUpdate", "init"};

    // If we aren't building standalone modules
    if(!standaloneModules) {
        // Because it won't be included in each
        // module, add runner to list of modules
        modules.push_back("runner");

        // **YUCK** this is kinda (ab)using standaloneModules for things it's not intended for but...
        // Show memory usage
        LOGI_CODE_GEN << "Host memory required for model: " << mem.getHostMBytes() << " MB";
        LOGI_CODE_GEN << "Device memory required for model: " << mem.getDeviceMBytes() << " MB";
        LOGI_CODE_GEN << "Zero-copy memory required for model: " << mem.getZeroCopyMBytes() << " MB";

        // Give warning of model requires more memory than device has
        if(mem.getDeviceBytes() > backend.getDeviceMemoryBytes()) {
            LOGW_CODE_GEN << "Model requires " << mem.getDeviceMBytes() << " MB of device memory but device only has " << backend.getDeviceMemoryBytes() / (1024 * 1024) << " MB";
        }

        // Output summary to log
        LOGI_CODE_GEN << "Merging model with " << model.getNeuronGroups().size() << " neuron groups and " << model.getSynapseGroups().size() << " synapse groups results in:";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedNeuronUpdateGroups().size() << " merged neuron update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedPresynapticUpdateGroups().size() << " merged presynaptic update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedPostsynapticUpdateGroups().size() << " merged postsynaptic update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseDynamicsGroups().size() << " merged synapse dynamics groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomUpdateGroups().size() << " merged custom update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomUpdateWUGroups().size() << " merged custom weight update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomUpdateTransposeWUGroups().size() << " merged custom weight transpose update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedNeuronInitGroups().size() << " merged neuron init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomUpdateInitGroups().size() << " merged custom update init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomWUUpdateDenseInitGroups().size() << " merged custom WU update dense init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseDenseInitGroups().size() << " merged synapse dense init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseConnectivityInitGroups().size() << " merged synapse connectivity init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseSparseInitGroups().size() << " merged synapse sparse init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedCustomWUUpdateSparseInitGroups().size() << " merged custom WU update sparse init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedNeuronSpikeQueueUpdateGroups().size() << " merged neuron spike queue update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseDendriticDelayUpdateGroups().size() << " merged synapse dendritic delay update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseConnectivityHostInitGroups().size() << " merged synapse connectivity host init groups";
    }

    // Return list of modules
    return std::make_pair(modules, mem);
}
