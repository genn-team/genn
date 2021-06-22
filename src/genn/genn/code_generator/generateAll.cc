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
//--------------------------------------------------------------------------
bool shouldRebuildModel(const filesystem::path &outputPath, const boost::uuids::detail::sha1::digest_type &hashDigest)
{
    try
    {
        // Open file
        std::ifstream is((outputPath / "model.sha").str());

        // Throw exceptions in case of all errors
        is.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);

        // Read previous hash digest as hex
        boost::uuids::detail::sha1::digest_type previousHashDigest; 
        is >> std::hex;
        for(auto &d : previousHashDigest) {
            is >> d;
        }
        
        // If hash matches
        if(previousHashDigest == hashDigest) {
            LOGI_CODE_GEN << "Model unchanged - skipping code generation";
            return false;
        }
        else {
            LOGI_CODE_GEN << "Model changed - re-generating code";
        }
    }
    catch(const std::ios_base::failure&) {
        LOGD_CODE_GEN << "Unable to read previous model hash - re-generating code";
    }

    // Open file
    std::ofstream os((outputPath / "model.sha").str());
    
    // Write digest as hex with each word seperated by a space
    os << std::hex;
    for(const auto d : hashDigest) {
        os << d << " ";
    }

    return true;
}
}   // Anonymous namespace

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
std::pair<std::vector<std::string>, CodeGenerator::MemAlloc> CodeGenerator::generateAll(const ModelSpecInternal &model, const BackendBase &backend,
                                                                                        const filesystem::path &sharePath, const filesystem::path &outputPath)
{
    // Create directory for generated code
    filesystem::create_directory(outputPath);

    // Create merged model
    ModelSpecMerged modelMerged(model, backend);
    
    // **HACK**
    MemAlloc mem = MemAlloc::zero();

    // If model should be rebuilt
    const auto hashDigest = modelMerged.getHashDigest(backend);
    if(shouldRebuildModel(outputPath, hashDigest)) {
        // Generate modules
        //**NOTE** memory spaces are given out on a first-come, first-serve basis so the modules should be in preferential order
        generateRunner(outputPath, modelMerged, backend);
        generateSynapseUpdate(outputPath, modelMerged, backend);
        generateNeuronUpdate(outputPath, modelMerged, backend);
        generateCustomUpdate(outputPath, modelMerged, backend);
        generateInit(outputPath, modelMerged, backend);

        // Generate support code module if the backend supports namespaces
        if(backend.supportsNamespace()) {
            generateSupportCode(outputPath, modelMerged);
        }

        // Get list of files to copy into generated code
        const auto backendSharePath = sharePath / "backends";
        const auto filesToCopy = backend.getFilesToCopy(modelMerged);
        const auto absOutputPath = outputPath.make_absolute();
        for(const auto &f : filesToCopy) {
            copyFile(f, backendSharePath, absOutputPath);
        }

        // Show memory usage
        LOGI_CODE_GEN << "Host memory required for model: " << mem.getHostMBytes() << " MB";
        LOGI_CODE_GEN << "Device memory required for model: " << mem.getDeviceMBytes() << " MB";
        LOGI_CODE_GEN << "Zero-copy memory required for model: " << mem.getZeroCopyMBytes() << " MB";

        // Give warning of model requires more memory than device has
        if(mem.getDeviceBytes() > backend.getDeviceMemoryBytes()) {
            LOGW_CODE_GEN << "Model requires " << mem.getDeviceMBytes() << " MB of device memory but device only has " << backend.getDeviceMemoryBytes() / (1024 * 1024) << " MB";
        }
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

    // Return list of modules and memory usage
    const std::vector<std::string> modules = {"customUpdate", "neuronUpdate", "synapseUpdate", "init", "runner"};
    return std::make_pair(modules, mem);
}
