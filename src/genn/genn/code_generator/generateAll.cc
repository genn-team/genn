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
#include "modelSpecMerged.h"

// Code generator includes
#include "code_generator/codeStream.h"
#include "code_generator/generateInit.h"
#include "code_generator/generateMPI.h"
#include "code_generator/generateNeuronUpdate.h"
#include "code_generator/generateSupportCode.h"
#include "code_generator/generateSynapseUpdate.h"
#include "code_generator/generateRunner.h"

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

    ModelSpecMerged mergedModel(model);

    // Generate modules
    generateNeuronUpdate(neuronUpdate, mergedModel, backend, standaloneModules);
    generateSynapseUpdate(synapseUpdate, model, backend, standaloneModules);
    generateInit(init, model, backend, standaloneModules);
    auto mem = generateRunner(definitions, definitionsInternal, runner, model, backend, 0);
    generateSupportCode(supportCode, model);

    // Create basic list of modules
    std::vector<std::string> modules = {"neuronUpdate", "synapseUpdate", "init"};

#ifdef MPI_ENABLE
    std::ofstream mpiStream((outputPath / "mpi.cc").str());
    CodeStream mpi(mpiStream);

    generateMPI(mpi, model, backend, standaloneModules);

    // Add MPI module
    modules.push_back("mpi");
#endif

    // If we aren't building standalone modules
    if(!standaloneModules) {
        // Because it won't be included in each
        // module, add runner to list of modules
        modules.push_back("runner");

        // **YUCK** this is kinda (ab)using standaloneModules for things it's not intended for but...
        // Show memory usage
        LOGI << "Host memory required for model: " << mem.getHostMBytes() << " MB";
        LOGI << "Device memory required for model: " << mem.getDeviceMBytes() << " MB";
        LOGI << "Zero-copy memory required for model: " << mem.getZeroCopyMBytes() << " MB";

        // Give warning of model requires more memory than device has
        if(mem.getDeviceBytes() > backend.getDeviceMemoryBytes()) {
            LOGW << "Model requires " << mem.getDeviceMBytes() << " MB of device memory but device only has " << backend.getDeviceMemoryBytes() / (1024 * 1024) << " MB";
        }
    }

    // Return list of modules
    return modules;
}
