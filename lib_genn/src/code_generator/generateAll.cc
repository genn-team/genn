#include "code_generator/generateAll.h"

// Standard C++ includes
#include <fstream>
#include <string>
#include <vector>

// Third party includes
#include "path.h"

// GeNN includes
#include "codeStream.h"

// Code generator includes
#include "code_generator/generateInit.h"
#include "code_generator/generateNeuronUpdate.h"
#include "code_generator/generateSynapseUpdate.h"
#include "code_generator/generateRunner.h"

//--------------------------------------------------------------------------
// CodeGenerator
//--------------------------------------------------------------------------
std::vector<std::string> CodeGenerator::generateAll(const NNmodel &model, const BackendBase &backend, const filesystem::path &outputPath)
{
    // Create directory for generated code
    filesystem::create_directory(outputPath);

    // Open output file streams for generated code files
    std::ofstream definitionsStream((outputPath / "definitions.h").str());
    std::ofstream neuronUpdateStream((outputPath / "neuronUpdate.cc").str());
    std::ofstream synapseUpdateStream((outputPath / "synapseUpdate.cc").str());
    std::ofstream initStream((outputPath / "init.cc").str());
    std::ofstream runnerStream((outputPath / "runner.cc").str());

    // Wrap output file streams in CodeStreams for formatting
    CodeStream definitions(definitionsStream);
    CodeStream neuronUpdate(neuronUpdateStream);
    CodeStream synapseUpdate(synapseUpdateStream);
    CodeStream init(initStream);
    CodeStream runner(runnerStream);

    // Generate modules
    generateNeuronUpdate(neuronUpdate, model, backend);
    generateSynapseUpdate(synapseUpdate, model, backend);
    generateInit(init, model, backend);
    generateRunner(definitions, runner, model, backend, 0);

    // Return names of generated modules
    return {"neuronUpdate", "synapseUpdate", "init", "runner"};
}