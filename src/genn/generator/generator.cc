// Standard C++ includes
#include <fstream>
#include <thread>

// PLOG includes
#include <plog/Appenders/ConsoleAppender.h>

// Filesystem includes
#include "path.h"

// GeNN includes
#include "logging.h"
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/generateMakefile.h"
#include "code_generator/generateNMakefile.h"
#include "code_generator/generateModules.h"
#include "code_generator/generateMSBuild.h"
#include "code_generator/modelSpecMerged.h"

// GeNN runtime includes
#include "runtime/runtime.h"

// Include backend
#include "optimiser.h"

// Declare global GeNN preferences
using namespace GeNN;
using namespace GeNN::CodeGenerator::BACKEND_NAMESPACE;
Preferences GENN_PREFERENCES;

// Include model
#include MODEL

int main(int argc,     //!< number of arguments; expected to be 3
         char *argv[]) //!< Arguments; expected to contain the genn directory and the target directory for code generation.
{
    try
    {
        if (argc != 4) {
            std::cerr << "usage: generator <genn dir> <target dir> <force rebuild>" << std::endl;
            return EXIT_FAILURE;
        }

        const filesystem::path gennPath(argv[1]);
        const filesystem::path targetPath(argv[2]);
        const bool forceRebuild = (std::stoi(argv[3]) != 0);

        // Initialise logging, appending all to console
        plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
        Logging::init(GENN_PREFERENCES.gennLogLevel, GENN_PREFERENCES.codeGeneratorLogLevel, 
                      GENN_PREFERENCES.transpilerLogLevel, GENN_PREFERENCES.runtimeLogLevel, 
                      &consoleAppender, &consoleAppender, &consoleAppender, &consoleAppender);

        // Create model
        // **NOTE** casting to external-facing model to hide model's internals
        ModelSpecInternal model;
        modelDefinition(static_cast<ModelSpec&>(std::ref(model)));

        // Finalize model
        model.finalise();

        // Determine code generation path
        const filesystem::path outputPath = targetPath / (model.getName() + "_CODE");

        // Create output path
        filesystem::create_directory(outputPath);

        // Create backend
        auto backend = Optimiser::createBackend(model, outputPath,
                                                GENN_PREFERENCES.backendLogLevel, &consoleAppender,
                                                GENN_PREFERENCES);

        // Create merged model and generate code
        CodeGenerator::ModelSpecMerged modelMerged(backend, model);
        const auto moduleNames = CodeGenerator::generateAll(modelMerged, backend, 
                                                            outputPath, forceRebuild);
        std::string buildCommand;
#ifdef _WIN32
        if(backend.shouldUseNMakeBuildSystem()) {
            // Create NMake file to compile and link all generated modules
            {
                std::ofstream nmake((outputPath / "Makefile").str());
                CodeGenerator::generateNMakefile(nmake, backend, moduleNames);
            }
            
            buildCommand = "cd \"" + outputPath.str() + "\" & nmake /NOLOGO";
        }
        else {
            // Create MSBuild project to compile and link all generated modules
            {
                std::ofstream msbuild((outputPath / "runner.vcxproj").str());
                CodeGenerator::generateMSBuild(msbuild, backend, moduleNames);
            }
            
            // Generate command to build using msbuild
            const std::string config = GENN_PREFERENCES.debugCode ? "Debug" : "Release";
            buildCommand = "msbuild /m /p:Configuration=" + config + " /verbosity:quiet \"" + (outputPath / "runner.vcxproj").str() + "\"";
        }
#else
        // Create makefile to compile and link all generated modules
        {
            std::ofstream makefile((outputPath / "Makefile").str());
            CodeGenerator::generateMakefile(makefile, backend, moduleNames);
        }

        // Generate command to build using make, using as many threads as possible
        const unsigned int numThreads = std::thread::hardware_concurrency();
        buildCommand = "make -C \"" + outputPath.str() + "\" -j " + std::to_string(numThreads);
#endif

        // Execute build command
        const int retval = system(buildCommand.c_str());
        if (retval != 0) {
            throw std::runtime_error("Building generated code with call:'" + buildCommand + "' failed with return value:" + std::to_string(retval));
        }
        
        // Create runtime and simulate
        Runtime::Runtime runtime(targetPath, modelMerged, backend);
        simulate(static_cast<ModelSpec&>(std::ref(model)), runtime);

    }
    catch(const std::exception &exception)
    {
        std::cerr << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
