/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/
#include <cmath>
#include <iostream>
#include <fstream>

// Userproject includes
#include "timer.h"

#include "PoissonIzh_CODE/definitions.h"

#include "sizes.h"

// other stuff:
#define REPORT_TIME 1000.0
#define SYN_OUT_TIME 2000.0

#define TOTAL_TIME 5000

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "usage: PoissonIzhSim <basename>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string outLabel = argv[1];
    const std::string outDir = "../" + outLabel + "_output";

    std::cerr << "# DT " << DT << std::endl;
    std::cerr << "# REPORT_TIME " << REPORT_TIME << std::endl;
    std::cerr << "# SYN_OUT_TIME " << SYN_OUT_TIME << std::endl;
    std::cerr << "# TOTAL_TIME " << TOTAL_TIME << std::endl;

    //-----------------------------------------------------------------
    // build the neuronal circuitry
    allocateMem();
    initialize();
    initializeSparse();

    std::cerr << "# neuronal circuitry built, start computation ..."  << std::endl;

    unsigned int sumPN = 0;
    unsigned int sumIzh1 = 0;
    double elapsedTime = 0.0;
    {
        //AnalogueRecorder<scalar> izhVoltage(outLabel + "_Vm", VIzh1, _NIzh);
        TimerAccumulate timer(elapsedTime);

        while(t < TOTAL_TIME) {
            stepTime();

            pullVIzh1FromDevice();
            pullPNCurrentSpikesFromDevice();
            pullIzh1CurrentSpikesFromDevice();

            // Sum spikes
            sumPN += getPNCurrentSpikeCount();
            sumIzh1 += getIzh1CurrentSpikeCount();

            // Record voltages
            //izhVoltage.record(t);

            if(fmod(t, REPORT_TIME) < 1e-3f) {
                std::cout << "time " << t << std::endl;
            }

        }
    }

    std::cout << sumPN << " Poisson spikes evoked spikes on " << sumIzh1 << " Izhikevich neurons in " << elapsedTime << " seconds." << std::endl;

    if(_TIMING) {
        std::cout << "Initialization time:" << initTime << "s" << std::endl;
        std::cout << "Sparse initialization time:" << initSparseTime << "s" << std::endl;
        std::cout << "Neuron update time:" << neuronUpdateTime << "s" << std::endl;
        std::cout << "Presynaptic update time:" << presynapticUpdateTime << "s" << std::endl;
    }
    return 0;
}
