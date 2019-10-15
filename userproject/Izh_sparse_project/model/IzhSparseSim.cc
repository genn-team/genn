/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu

   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/
// Standard C++ includes
#include <iostream>
#include <fstream>

#include <cmath>

#include "../include/spikeRecorder.h"

// Model includes
#include "IzhSparse_CODE/definitions.h"

#include "sizes.h"


//----------------------------------------------------------------------
// other stuff:
#define REPORT_TIME 5000.0
#define TOTAL_TIME 1000.0

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "usage: Izh_sparse_sim <basename> <CPU=0, GPU=1> \n");
        return EXIT_FAILURE;
    }

    const std::string outLabel = argv[1];
    const std::string outDir = "../" + outLabel + "_output";

    // Calculate number of neurons in each population
    const unsigned int nExc = (unsigned int)ceil(4.0 * _NNeurons / 5.0);
    const unsigned int nInh = _NNeurons - nExc;

    allocateMem();
    initialize();

    // Download GPU-initialized 'b' parameter values from device
    pullbPInhFromDevice();

    // Manually initially U of inhibitory population
    std::transform(&bPInh[0], &bPInh[nInh], &UPInh[0],
                   [](float b){ return b * -65.0f; });

    initializeSparse();

    std::cout << "# neuronal circuitry built, start computation ... \n\n" << std::endl;

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation


    std::cout << "# DT " << DT << std::endl;
    std::cout << "# T_REPORT_TME " << REPORT_TIME << std::endl;
    std::cout << "# TOTAL_TME " << TOTAL_TIME << std::endl;

    std::cout << "# We are running with fixed time step " <<  DT << std::endl;;

    SpikeRecorder<> excSpikes(&getPExcCurrentSpikes, &getPExcCurrentSpikeCount, outDir + "/" + outLabel + "_exc_st");
    SpikeRecorder<> inhSpikes(&getPInhCurrentSpikes, &getPInhCurrentSpikeCount, outDir + "/" + outLabel + "_inh_st");

    while(t < TOTAL_TIME) {
        stepTime();

        pullPExcCurrentSpikesFromDevice();
        pullPInhCurrentSpikesFromDevice();

        excSpikes.record(t);
        inhSpikes.record(t);

    }

    if(_TIMING) {
        std::cout << "Initialization time:" << initTime << "s" << std::endl;
        std::cout << "Initialization sparse time:" << initSparseTime << "s" << std::endl;
        std::cout << "Neuron update time:" << neuronUpdateTime << "s" << std::endl;
        std::cout << "Presynaptic update time:" << presynapticUpdateTime << "s" << std::endl;
    }
    return EXIT_SUCCESS;
}


