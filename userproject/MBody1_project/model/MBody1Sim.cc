/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Science
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file userproject/MBody1_project/model/classol_sim.cc

\brief Main entry point for the classol (CLASSification in OLfaction) model simulation. Provided as a part of the complete example of simulating the MBody1 mushroom body model. 
*/
//--------------------------------------------------------------------------
// Standard C includes
#include <cmath>

// Userproject includes
#include "spikeRecorder.h"
#include "timer.h"

#include "sizes.h"

#include "MBody1_CODE/definitions.h"

//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 10000.0
#define SYN_OUT_TME 20000.0


// reset input every 100ms
#define PAT_TIME 100.0

// pattern goes off after 10ms
#define PATFTIME 10.0

#define TOTAL_TME 5000.0

//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the simulation of the MBody1 model network.
*/
//--------------------------------------------------------------------------


int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "usage: MBody1 <basename>" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string outLabel = argv[1];
    const std::string outDir = "../" + outLabel + "_output";

    const unsigned long long patSetTimeSteps = (unsigned long long)std::round(PAT_TIME / DT);
    const unsigned long long patFireTimeSteps = (unsigned long long)std::round(PATFTIME / DT);
    const unsigned long long numPatterns = 100;
    const scalar baseRateHz = 0.2f;

    std::cout << "# DT " <<  DT << std::endl;
    std::cout << "# T_REPORT_TME " <<  T_REPORT_TME << std::endl;
    std::cout << "# SYN_OUT_TME " <<  SYN_OUT_TME << std::endl;
    std::cout << "# PATFTIME " <<  PATFTIME << std::endl;
    std::cout << "# patFireTimesteps " <<  patFireTimeSteps << std::endl;
    std::cout << "# PAT_TIME " <<  PAT_TIME << std::endl;
    std::cout << "# patSetTimesteps " <<  patSetTimeSteps << std::endl;
    std::cout << "# TOTAL_TME " <<  TOTAL_TME << std::endl;

    //-----------------------------------------------------------------
    // build the neuronal circuitry
    allocateMem();

    initialize();

    // Pull initialised KCDN conductances from device and use to calculate gRawKCDN
    pullgKCDNFromDevice();
    std::transform(&gKCDN[0], &gKCDN[_NKC*_NDN], &gRawKCDN[0],
                   [](scalar g)
                   {
                       const double tmp = (double)g / 0.015 * 2.0;
                       return (scalar)(0.5 * log(tmp / (2.0 - tmp)) / 33.33 + 0.0075);
                   });

    initializeSparse();

    // Allocate extra extra global parameter for patterns
    allocatefiringProbPN(_NAL * (1 + numPatterns));

    // Fill first _NAL entries with baserate
    const scalar baseFiringProb = (baseRateHz / 1000.0) * DT;
    std::fill_n(&firingProbPN[0], _NAL, baseFiringProb);


    {
        Timer a("%% Reading input patterns: ");

        // Open input pattern rates file
        std::ifstream patternRatesFile(outDir + "/" + outLabel + ".inpat", std::ios::binary);
        if(!patternRatesFile.good()) {
            std::cerr << "Cannot open input patterns file" << std::endl;
            return EXIT_FAILURE;
        }

        // Read pattern rates from disk
        std::vector<double> patternRates(_NAL * numPatterns);
        patternRatesFile.read(reinterpret_cast<char*>(patternRates.data()), _NAL * numPatterns * sizeof(double));

         // Check block was read succesfully
        if((size_t)patternRatesFile.gcount() != (_NAL * numPatterns * sizeof(double))) {
            std::cerr << "Unexpected end of patterns file" << std::endl;
            return EXIT_FAILURE;
        }

        // Transform each rate in hz into KHz and thus into mean spikes per timestep
        std::transform(patternRates.cbegin(), patternRates.cend(), &firingProbPN[_NAL],
                       [](double rateHz){ return (rateHz / 1000.0) * DT; });
    }

    // Upload firing probabilities to GPU
    pushfiringProbPNToDevice(_NAL * (1 + numPatterns));

    std::cout << "# neuronal circuitry built, start computation ... " << std::endl;

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation

    std::cout << "# We are running with fixed time step " << DT << std::endl;
    //timer.startTimer();
    SpikeRecorder pnSpikes(outDir + "/" + outLabel + ".pn.st", glbSpkCntPN, glbSpkPN);
    SpikeRecorder lhiSpikes(outDir + "/" + outLabel + ".lhi.st", glbSpkCntLHI, glbSpkLHI);

#ifdef DELAYED_SYNAPSES
    SpikeRecorderDelay kcSpikes(_NKC, spkQueuePtrKC, outDir + "/" + outLabel + ".kc.st", glbSpkCntKC, glbSpkKC);
    SpikeRecorderDelay dnSpikes(_NDN, spkQueuePtrDN, outDir + "/" + outLabel + ".dn.st", glbSpkCntDN, glbSpkDN);
#else
    SpikeRecorder kcSpikes(outDir + "/" + outLabel + ".kc.st", glbSpkCntKC, glbSpkKC);
    SpikeRecorder dnSpikes(outDir + "/" + outLabel + ".dn.st", glbSpkCntDN, glbSpkDN);
#endif

    double simTime = 0.0;
    {
        TimerAccumulate a(simTime);

        while(t < TOTAL_TME) {
            if((iT % patSetTimeSteps) == 0) {
                const unsigned int pno = (iT / patSetTimeSteps) % numPatterns;
                offsetPN = (pno + 1) *_NAL;
            }
            if((iT % patSetTimeSteps) == patFireTimeSteps) {
                offsetPN = 0;
            }

            stepTime();

            // Copy current spikes from all populations from device
            copyCurrentSpikesFromDevice();

            // Record spikes
            pnSpikes.record(t);
            kcSpikes.record(t);
            lhiSpikes.record(t);
            dnSpikes.record(t);

            // report progress
            if(fmod(t, T_REPORT_TME) < 1e-3f) {
                std::cout << "time " << t << std::endl;
            }
        }
    }


    pullVDNFromDevice();
    std::cerr << "output files are created under the current directory." << std::endl;

    const unsigned int numNeurons = _NAL + _NKC + _NLHI + _NDN;
    std::cout << numNeurons << " neurons, " << pnSpikes.getSum() << " PN spikes, " << kcSpikes.getSum() << " KC spikes, " << lhiSpikes.getSum() << " LHI spikes, ";
    std::cout << dnSpikes.getSum() << " DN spikes, " << "simulation took " << simTime << " secs, VDN[0]=" << VDN[0] << " DT=" << DT << std::endl;

    if(_TIMING) {
        std::cout << "Initialization time:" << initTime << "s" << std::endl;
        std::cout << "Sparse initialization time:" << initSparseTime << "s" << std::endl;
        std::cout << "Neuron update time:" << neuronUpdateTime << "s" << std::endl;
        std::cout << "Presynaptic update time:" << presynapticUpdateTime << "s" << std::endl;
        std::cout << "Postsynaptic update time:" << postsynapticUpdateTime << "s" << std::endl;
    }

    // Free everything
    freefiringProbPN();
    freeMem();

    return 0;
}
