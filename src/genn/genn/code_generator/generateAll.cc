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
#include "code_generator/generateInit.h"
#include "code_generator/generateNeuronUpdate.h"
#include "code_generator/generateSupportCode.h"
#include "code_generator/generateSynapseUpdate.h"
#include "code_generator/generateRunner.h"
#include "code_generator/modelSpecMerged.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
std::vector<std::string> CodeGenerator::generateAll(const ModelSpecInternal &model, const BackendBase &backend,
                                                    const filesystem::path &outputPath, bool standaloneModules)
{
    // Create directory for generated code
    filesystem::create_directory(outputPath);

    // Open output file streams for generated code files
    std::ofstream definitionsStream((outputPath / "definitions.h").str());
    std::ofstream definitionsInternalStream((outputPath / "definitionsInternal.h").str());
    std::ofstream supportCodeStream((outputPath / "supportCode.h").str());
    std::ofstream neuronUpdateStream((outputPath / "neuronUpdate.cc").str());
    std::ofstream synapseUpdateStream((outputPath / "synapseUpdate.cc").str());
    std::ofstream initStream((outputPath / "init.cc").str());
    std::ofstream runnerStream((outputPath / "runner.cc").str());

    // Wrap output file streams in CodeStreams for formatting
    CodeStream definitions(definitionsStream);
    CodeStream definitionsInternal(definitionsInternalStream);
    CodeStream supportCode(supportCodeStream);
    CodeStream neuronUpdate(neuronUpdateStream);
    CodeStream synapseUpdate(synapseUpdateStream);
    CodeStream init(initStream);
    CodeStream runner(runnerStream);

    // Create merged model
    ModelSpecMerged modelMerged(model, backend);

    // Generate modules
    MergedStructData mergedStructData;
    auto mem = generateRunner(definitions, definitionsInternal, runner, mergedStructData, modelMerged, backend);
    generateNeuronUpdate(neuronUpdate, mergedStructData, modelMerged, backend);
    generateSynapseUpdate(synapseUpdate, mergedStructData, modelMerged, backend);
    generateInit(init, mergedStructData, modelMerged, backend);

    generateSupportCode(supportCode, modelMerged);

    // Create basic list of modules
    std::vector<std::string> modules = {"neuronUpdate", "synapseUpdate", "init"};

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
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedNeuronInitGroups().size() << " merged neuron init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseDenseInitGroups().size() << " merged synapse dense init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseConnectivityInitGroups().size() << " merged synapse connectivity init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseSparseInitGroups().size() << " merged synapse sparse init groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedNeuronSpikeQueueUpdateGroups().size() << " merged neuron spike queue update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseDendriticDelayUpdateGroups().size() << " merged synapse dendritic delay update groups";
        LOGI_CODE_GEN << "\t" << modelMerged.getMergedSynapseConnectivityHostInitGroups().size() << " merged synapse connectivity host init groups";
    }

    // Return list of modules
    return modules;
}
