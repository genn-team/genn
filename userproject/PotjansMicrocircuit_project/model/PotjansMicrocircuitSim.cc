// Standard C++ includes
#include <memory>
#include <random>
#include <vector>

// GeNN user project includes
#include "../include/sharedLibraryModel.h"
#include "../include/spikeRecorder.h"
#include "../include/timer.h"

// Model parameters
#include "PotjansMicrocircuitParams.h"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "usage: PotjansMicrocircuit <basename>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string outLabel = argv[1];
    const std::string outDir = "../" + outLabel + "_output";
    const unsigned int timesteps = round(Parameters::durationMs / Parameters::dtMs);
    
    SharedLibraryModel<float> model("./", "PotjansMicrocircuit");

    model.allocateMem();
    model.allocateRecordingBuffers(timesteps);
    model.initialize();
    model.initializeSparse();

    double recordS = 0.0;
    {
        Timer timer("Simulation:");

        // Loop through timesteps
        const unsigned int tenPercentTimestep = timesteps / 10;
        for(unsigned int i = 0; i < timesteps; i++) {
            // Indicate every 10%
            if((i % tenPercentTimestep) == 0) {
                std::cout << i / 100 << "%" << std::endl;
            }

            // Simulate
            model.stepTime();
        }
    }
    {
        TimerAccumulate timer(recordS);

        // Download recording data from device
        model.pullRecordingBuffersFromDevice();

        // Loop through populations
        for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
            for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
                // Get pointer to recording data
                const std::string name = Parameters::getPopulationName(layer, pop);
                const uint32_t *recordSpk = model.getArray<uint32_t>("recordSpk" + name);

                // Write to text file
                writeTextSpikeRecording(outDir + "/" + outLabel + "." + name + ".st", recordSpk, Parameters::getScaledNumNeurons(layer, pop), timesteps, Parameters::dtMs);
            }
        }
    }
    
    if(Parameters::measureTiming) {
        std::cout << "Timing:" << std::endl;
        std::cout << "\tInit:" << *model.getScalar<double>("initTime") * 1000.0 << std::endl;
        std::cout << "\tSparse init:" << *model.getScalar<double>("initSparseTime") * 1000.0 << std::endl;
        std::cout << "\tNeuron simulation:" << *model.getScalar<double>("neuronUpdateTime") * 1000.0 << std::endl;
        std::cout << "\tSynapse simulation:" << *model.getScalar<double>("presynapticUpdateTime") * 1000.0 << std::endl;
    }
    std::cout << "Record:" << recordS << "s" << std::endl;

    return EXIT_SUCCESS;
}
