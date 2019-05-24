// SpineML simulator includes
#include "simulator.h"

using namespace SpineMLSimulator;

int main(int argc, char *argv[])
{
    try
    {
        if(argc < 2) {
            throw std::runtime_error("Expected experiment XML file passed as arguments");
        }

        // Read min log severity from command line
        const plog::Severity minSeverity = (argc > 3) ? (plog::Severity)std::stoi(argv[3]) : plog::info;

        // Initialise log channels, appending all to console
        plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
        plog::init(minSeverity, &consoleAppender);

#ifdef _WIN32
        // Startup WinSock 2
        WSADATA wsaData;
        if(WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }

#endif  // _WIN32

        // Create simulator
        Simulator simulator(argv[1], argv[2]);

        const unsigned int numTimeSteps = simulator.calcNumTimesteps();
        LOGI << "Simulating for " << numTimeSteps << " " << simulator.getDT() << "ms timesteps";

        // Loop through time
        for(unsigned int i = 0; i < numTimeSteps; i++) {
            simulator.stepTime();
        }

        LOGI << "Applying input: " << simulator.getInputMs() << "ms, simulating:" << simulator.getSimulateMs() << "ms, logging:" << simulator.getLogMs() << "ms" << std::endl;
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
