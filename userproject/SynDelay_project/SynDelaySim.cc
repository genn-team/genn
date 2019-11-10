//--------------------------------------------------------------------------
//   Author:    James Turner
//  
//   Institute: Center for Computational Neuroscience and Robotics
//              University of Sussex
//              Falmer, Brighton BN1 9QJ, UK 
//  
//   email to:  J.P.Turner@sussex.ac.uk
//  
//--------------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <fstream>

// User project includes
#include "spikeRecorder.h"
#include "timer.h"

#include "SynDelay_CODE/definitions.h"


#define TOTAL_TIME 5000.0f
#define REPORT_TIME 1000.0f

/*====================================================================
  --------------------------- MAIN FUNCTION ----------------------------
  ====================================================================*/

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "usage: SynDelaySim <output label>" << std::endl;
        return EXIT_FAILURE;
    }

    allocateMem();
    initialize();
    initializeSparse();

    const std::string outLabel = argv[1];
    std::ofstream fileV;
    fileV.open((outLabel + "_Vm").c_str(), std::ios::out | std::ios::trunc);

    {
        SpikeRecorder<> inputSpikes(&getInputCurrentSpikes, &getInputCurrentSpikeCount, outLabel + "_input_st");
        SpikeRecorder<> interSpikes(&getInterCurrentSpikes, &getInterCurrentSpikeCount, outLabel + "_inter_st");
        SpikeRecorder<> outputSpikes(&getOutputCurrentSpikes, &getOutputCurrentSpikeCount, outLabel + "_output_st");

        std::cout << "# DT " << DT << std::endl;
        std::cout << "# TOTAL_TIME " << TOTAL_TIME << std::endl;
        std::cout << "# REPORT_TIME " << REPORT_TIME << std::endl;

        Timer timer("# done in ", outLabel + "_time");

        while(t < TOTAL_TIME) {
            stepTime();

            copyStateFromDevice();
            pullInputCurrentSpikesFromDevice();
            pullInterCurrentSpikesFromDevice();
            pullOutputCurrentSpikesFromDevice();

            fileV << t
                    << " " << VInput[0]
                    << " " << VInter[0]
                    << " " << VOutput[0]
                    << std::endl;

            inputSpikes.record(t);
            interSpikes.record(t);
            outputSpikes.record(t);

            if(fmod(t, REPORT_TIME) < 1e-3f) {
                std::cout << "time " << t << std::endl;
            }
        }
    }

    fileV.close();

    freeMem();

    return EXIT_SUCCESS;
}

