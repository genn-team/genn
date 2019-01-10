// Standard C++ includes
#include <fstream>

// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

// Filesystem includes
#include "path.h"

// GeNN includes
#include "modelSpec.h"

// GeNN code generator includes
#include "code_generator/generateAll.h"
#include "code_generator/generateMakefile.h"

// Include backend
#include "backend.h"

enum Log
{
    LogDefault,
    LogBackend,
    LogOptimiser,
};

// Declare global GeNN preferences
CodeGenerator::BACKEND_TYPE::Backend::Preferences GENN_PREFERENCES;

// Include model
#include MODEL

int main(int argc,     //!< number of arguments; expected to be 2
         char *argv[]) //!< Arguments; expected to contain the target directory for code generation.
    
{
    // Initialise log channels, appending all to console
    // **TODO** de-crud standard logger
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init<LogDefault>(plog::info, &consoleAppender);
    plog::init<LogBackend>(plog::info, &consoleAppender);
    plog::init<LogOptimiser>(plog::info, &consoleAppender);
    
    if (argc != 2) {
        LOGE << "usage: generateALL <target dir>";
        return EXIT_FAILURE;
    }
    
    // Create output path
    const filesystem::path outputPath(argv[1]);
    filesystem::create_directory(outputPath);

    // Create model
    NNmodel model;
    modelDefinition(model);
    
    // Finalize model
    model.finalize();

    int localHostID = 0;

#ifdef MPI_ENABLE
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &localHostID);
    cout << "MPI initialized - host ID:" << localHostID << endl;
#endif
    
    // Create backend
    auto backend = CodeGenerator::BACKEND_TYPE::Optimiser::createBackend(model, outputPath, localHostID, GENN_PREFERENCES);
    
    // Generate code
    const auto moduleNames = CodeGenerator::generateAll(model, backend, outputPath);

    // Create makefile to compile and link all generated modules
    std::ofstream makefile((outputPath / "Makefile").str());
    CodeGenerator::generateMakefile(makefile, backend, moduleNames);
    
#ifdef MPI_ENABLE
    MPI_Finalize();
    cout << "MPI finalized." << endl;
#endif
    return EXIT_SUCCESS;
}