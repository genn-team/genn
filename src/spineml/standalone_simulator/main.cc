// PLOG includes
#include <plog/Log.h>
#include <plog/Appenders/ConsoleAppender.h>

// CLI11 includes
#include "CLI11.hpp"

// SpineML simulator includes
#include "simulator.h"

using namespace SpineMLSimulator;

int main(int argc, char *argv[])
{
    try
    {
        CLI::App app{"SpineML simulator for GeNN"};

        std::string experimentFilename;
        std::string outputDirectory;
        unsigned int logLevel = plog::info;

        app.add_option("experiment,-e,--experiment", experimentFilename, "Experiment xml file")->required();
        app.add_option("output,-o,--output", outputDirectory, "Output directory for generated code");
        app.add_flag("--log-error{2},--log-warning{3},--log-info{4},--log-debug{5}", logLevel, "Verbosity of logging to show");

        CLI11_PARSE(app, argc, argv);

        // Initialise log channels, appending all to console
        plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
        plog::init((plog::Severity)logLevel, &consoleAppender);

#ifdef _WIN32
        // Startup WinSock 2
        WSADATA wsaData;
        if(WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }

#endif  // _WIN32

        // Create simulator
        Simulator simulator(experimentFilename,outputDirectory);

        const unsigned long long numTimeSteps = simulator.calcNumTimesteps();
        LOGI << "Simulating for " << numTimeSteps << " " << simulator.getDT() << "ms timesteps";

        // Loop through time
        for(unsigned long long i = 0; i < numTimeSteps; i++) {
            simulator.stepTime();
        }

        LOGI << "Applying input: " << simulator.getInputMs() << "ms, simulating:" << simulator.getSimulateMs() << "ms, logging:" << simulator.getLogMs() << "ms" << std::endl;

#ifdef _WIN32
        // Close down WinSock 2
        WSACleanup();
#endif
    }
    catch(const std::exception &exception)
    {
        LOGE << exception.what();
        return EXIT_FAILURE;
    }
    catch(...)
    {
        LOGE << "Unknown exception";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
