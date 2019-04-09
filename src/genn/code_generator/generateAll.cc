#include "code_generator/generateAll.h"

// Standard C++ includes
#include <fstream>
#include <string>
#include <vector>

// Third party includes
#include "path.h"

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

    // Generate modules
    generateNeuronUpdate(neuronUpdate, model, backend, standaloneModules);
    generateSynapseUpdate(synapseUpdate, model, backend, standaloneModules);
    generateInit(init, model, backend, standaloneModules);
    generateRunner(definitions, definitionsInternal, runner, model, backend, 0);
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

    // If we aren't building standalone modules, in which case it
    // will be includes in each one, add runner to list of modules
    if(!standaloneModules) {
        modules.push_back("runner");
    }

    // Return list of modules
    return modules;
}
