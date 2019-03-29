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

#include "../include/analogue_recorder.h"
#include "../include/timer.h"

#include "OneComp_CODE/definitions.h"

#include "sizes.h"

//----------------------------------------------------------------------
// other stuff:
#define REPORT_TIME 100.0
#define TOTAL_TIME 5000.0

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "usage: OneCompSim <output label>" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string outLabel = argv[1];

    std::cerr << "# DT " << DT << std::endl;
    std::cerr << "# REPORT_TIME " << REPORT_TIME << std::endl;
    std::cerr << "# TOTAL_TIME " << TOTAL_TIME << std::endl;

    FILE *osf= fopen((outLabel + "_v").c_str(),"w");

    //-----------------------------------------------------------------
    // build the neuronal circuitry
    allocateMem();
    initialize();
    initializeSparse();

    std::cerr << "# neuronal circuitry built, start computation ..." << std::endl;

    {
        AnalogueRecorder<scalar> izhVoltage(outLabel + "_Vm", VIzh1, _NC1);
        Timer timer("# done in ", outLabel + "_time");

        while(t < TOTAL_TIME) {
            stepTime();

            // Download spikes and voltage from device
            pullIzh1CurrentSpikesFromDevice();
            pullVIzh1FromDevice();

            // Record voltages
            izhVoltage.record(t);

            if(fmod(t, REPORT_TIME) < 1e-3f) {
                std::cout << "time " << t << std::endl;
            }
        }
    }

    return EXIT_SUCCESS;
}
