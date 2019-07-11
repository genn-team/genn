// Standard C++ includes
#include <fstream>

#ifdef _WIN32
#include <Objbase.h>
#endif

// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

// Filesystem includes
#include "path.h"

// GeNN includes
#include "modelSpecInternal.h"

// GeNN code generator includes
#include "code_generator/generateAll.h"
#include "code_generator/generateMakefile.h"
#include "code_generator/generateMSBuild.h"

// Include backend
#include "optimiser.h"

// Declare global GeNN preferences
using namespace CodeGenerator::BACKEND_NAMESPACE;
Preferences GENN_PREFERENCES;

// Include model
#include MODEL

int main(int argc,     //!< number of arguments; expected to be 2
         char *argv[]) //!< Arguments; expected to contain the target directory for code generation.
{
    try
    {
        if (argc != 2) {
            std::cerr << "usage: generator <target dir>";
            return EXIT_FAILURE;
        }
        
        const filesystem::path targetPath(argv[1]);

        // Create code generation path
        int localHostID = 0;
#ifdef MPI_ENABLE
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &localHostID);
        std::cout << "MPI initialized - host ID:" << localHostID;
#endif

        // Create model
        // **NOTE** casting to external-facing model to hide model's internals
        ModelSpecInternal model;
        modelDefinition(static_cast<ModelSpec&>(std::ref(model)));
        
        // Initialise logging, appending all to console
        plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;

        // If there isn't already a plog instance, initialise one
        if(plog::get() == nullptr) {
            plog::init(GENN_PREFERENCES.logLevel, &consoleAppender);
        }
        // Otherwise, set it's max severity from GeNN preferences
        else {
            plog::get()->setMaxSeverity(GENN_PREFERENCES.logLevel);
        }

        // Finalize model
        model.finalize();

        // Create code generation path
#ifdef MPI_ENABLE
        const filesystem::path outputPath = targetPath / (model.getName() + "_" + std::to_string(localHostID) + "_CODE");
#else
        const filesystem::path outputPath = targetPath / (model.getName() + "_CODE");
#endif
        // Create output path
        filesystem::create_directory(outputPath);

        // Create backend
        auto backend = Optimiser::createBackend(model, outputPath, localHostID, GENN_PREFERENCES);
        
        // Generate code
        const auto moduleNames = CodeGenerator::generateAll(model, backend, outputPath);

#ifdef _WIN32
        // If runner GUID file doesn't exist
        const filesystem::path projectGUIDFilename = targetPath / "runner_guid.txt";
        std::string projectGUIDString;
        if(!projectGUIDFilename.exists()) {
            // Create a new GUID for project
            GUID guid;
            if(::CoCreateGuid(&guid) != S_OK) {
                LOGE << "Unable to generate project GUID";
                return EXIT_FAILURE;
            }
            
            // Write GUID to string stream
            std::stringstream projectGUIDStream;
            projectGUIDStream << std::uppercase << std::hex << std::setfill('0');
            projectGUIDStream << std::setw(8)<< guid.Data1 << '-';
            projectGUIDStream << std::setw(4) << guid.Data2 << '-';
            projectGUIDStream << std::setw(4) << guid.Data3 << '-';
            projectGUIDStream << std::setw(2) << static_cast<short>(guid.Data4[0]) << std::setw(2) << static_cast<short>(guid.Data4[1]) << '-';
            projectGUIDStream << static_cast<short>(guid.Data4[2]) << static_cast<short>(guid.Data4[3]) << static_cast<short>(guid.Data4[4]) << static_cast<short>(guid.Data4[5]) << static_cast<short>(guid.Data4[6]) << static_cast<short>(guid.Data4[7]);
            
            // Use result as project GUID string
            projectGUIDString = projectGUIDStream.str();
            LOGI << "Generated new project GUID:" << projectGUIDString;
            
            // Write GUID to project GUID file
            std::ofstream projectGUIDFile(projectGUIDFilename.str());
            projectGUIDFile << projectGUIDString << std::endl;
        }
        // Otherwise
        else {
            // Read GUID from project GUID file
            std::ifstream projectGUIDFile(projectGUIDFilename.str());
            std::getline(projectGUIDFile, projectGUIDString);
            LOGI << "Using previously generated project GUID:" << projectGUIDString;
        }
        // Create MSBuild project to compile and link all generated modules
        std::ofstream makefile((outputPath / "runner.vcxproj").str());
        CodeGenerator::generateMSBuild(makefile, backend, projectGUIDString, moduleNames);
#else
        // Create makefile to compile and link all generated modules
        std::ofstream makefile((outputPath / "Makefile").str());
        CodeGenerator::generateMakefile(makefile, backend, moduleNames);
#endif

#ifdef MPI_ENABLE
        MPI_Finalize();
        std::cout << "MPI finalized";
#endif
    }
    catch(const std::exception &exception)
    {
        std::cerr << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
