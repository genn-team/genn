#include "definitions.h"
#include <fstream>
#include <iostream>

int main()
{
    allocateMem();
    std::cout << "Initialising" << std::endl;
    initialize();
    initializeSparse();

    std::cout << "Simulating" << std::endl;
    std::ofstream stream("spikes.csv");
    while(t < 1000.0f) {
        stepTime();
        pullExcCurrentSpikesFromDevice();
        pullInhCurrentSpikesFromDevice();

        for(unsigned int i = 0; i < spikeCount_Exc; i++) {
            stream << t << ", " << spike_Exc[i] << std::endl;
        }
        for(unsigned int i = 0; i < spikeCount_Inh; i++) {
            stream << t << ", " << 8000 + spike_Inh[i] << std::endl;
        }
    }

    return EXIT_SUCCESS;
}