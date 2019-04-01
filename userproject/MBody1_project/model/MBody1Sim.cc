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


#include "hr_time.h"
#include "modelSpec.h"

#include "map_classol.h"

#include "sizes.h"

#include "MBody1_CODE/definitions.h"

//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 10000.0
#define SYN_OUT_TME 20000.0


// reset input every 100 steps == 50ms
#define PAT_TIME 100.0

// pattern goes off at 2 steps == 1 ms
#define PATFTIME 1.5


#define TOTAL_TME 5000.0




//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the simulation of the MBody1 model network.
*/
//--------------------------------------------------------------------------


int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: classol_sim <basename> <CPU=0, GPU=1> \n");
        return 1;
    }
    int which= atoi(argv[2]);
    string OutDir = string(argv[1]) +"_output";
    string name;
    name= OutDir+ "/"+ argv[1] + ".time";
    FILE *timef= fopen(name.c_str(),"a");

    const int patSetTime= (int) (PAT_TIME/DT);
    const int patFireTime= (int) (PATFTIME/DT);
    fprintf(stdout, "# DT %f \n", DT);
    fprintf(stdout, "# T_REPORT_TME %f \n", T_REPORT_TME);
    fprintf(stdout, "# SYN_OUT_TME %f \n",  SYN_OUT_TME);
    fprintf(stdout, "# PATFTIME %f \n", PATFTIME);
    fprintf(stdout, "# patFireTime %d \n", patFireTime);
    fprintf(stdout, "# PAT_TIME %f \n", PAT_TIME);
    fprintf(stdout, "# patSetTime %d \n", patSetTime);
    fprintf(stdout, "# TOTAL_TME %f \n", TOTAL_TME);

    name= OutDir+ "/"+ argv[1] + ".out.Vm";
    FILE *osf= fopen(name.c_str(),"w");
    name= OutDir+ "/"+ argv[1] + ".out.st";
    FILE *osf2= fopen(name.c_str(),"w");

#ifdef TIMING
    name= OutDir+ "/"+ argv[1] + ".timingprofile";
    FILE *timeros= fopen(name.c_str(),"w");
    double tme;
#endif

    //-----------------------------------------------------------------
    // build the neuronal circuitry
    classol locust(patSetTime, patFireTime);
    CStopWatch timer;
#ifdef TIMING
    timer.startTimer();
#endif

    fprintf(stdout, "# reading PN-KC synapses ... \n");
    name= OutDir+ "/"+ argv[1] + ".pnkc";
    FILE *f= fopen(name.c_str(),"rb");
    locust.read_pnkcsyns(f);
    fclose(f);

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Reading PN-KC synapses: %f \n", tme);
    timer.startTimer();
#endif

    fprintf(stdout, "# reading PN-LHI synapses ... \n");
    name= OutDir+ "/"+ argv[1] + ".pnlhi";
    f= fopen(name.c_str(), "rb");
    locust.read_pnlhisyns(f);
    fclose(f);

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Reading PN-LHI synapses: %f \n", tme);
    timer.startTimer();
#endif

    fprintf(stdout, "# reading KC-DN synapses ... \n");
    name= OutDir+ "/"+ argv[1] + ".kcdn";
    f= fopen(name.c_str(), "rb");
    locust.read_kcdnsyns(f);

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Reading KC-DN synapses: %f \n", tme);
    timer.startTimer();
#endif

    fprintf(stdout, "# reading input patterns ... \n");
    name= OutDir+ "/"+ argv[1] + ".inpat";
    f= fopen(name.c_str(), "rb");
    locust.read_input_patterns(f);
    fclose(f);

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Reading input patterns: %f \n", tme);
    timer.startTimer();
#endif

    locust.generate_baserates();
#ifndef CPU_ONLY
    if (which == GPU) {
        locust.allocate_device_mem_patterns();
    }
#endif
    locust.init(which);         // this includes copying g's for the GPU version

#ifdef TIMING
    timer.stopTimer();
    tme= timer.getElapsedTime();
    fprintf(timeros, "%% Initialisation: %f \n", tme);
#endif

    fprintf(stdout, "# neuronal circuitry built, start computation ... \n\n");

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation

    fprintf(stdout, "# We are running with fixed time step %f \n", DT);
    t= 0.0;
    iT= 0;
    int done= 0;
    float last_t_report=  t;
    timer.startTimer();

    while (!done)
    {
        if (which == GPU) {
#ifndef CPU_ONLY
            locust.runGPU(DT); // run next batch
            //locust.getSpikeNumbersFromGPU();
            locust.getSpikesFromGPU();
#endif
        }
        else {
            locust.runCPU(DT); // run next batch
        }

    #ifdef TIMING
        fprintf(timeros, "%f %f %f \n", neuron_tme, synapse_tme, learning_tme);
    #endif
        locust.sum_spikes();
        locust.output_spikes(osf2, which);

        /*   fprintf(osf, "%f ", t);
            for (int i= 0; i < 100; i++) {
                fprintf(osf, "%f ", VDN[i]);
            }
            fprintf(osf,"\n");
        */
        // report progress
        if (t - last_t_report >= T_REPORT_TME)
        {
            fprintf(stdout, "time %f \n", t);
            last_t_report= t;
        }
        done= (t >= TOTAL_TME);
    }

    timer.stopTimer();
#ifndef CPU_ONLY
    if (which == GPU) {
        pullDNStateFromDevice();
    }
#endif
    cerr << "output files are created under the current directory." << endl;
    const unsigned int numNeurons = _NAL + _NMB + _NLHI + _NLB;
    fprintf(timef, "%d %u %u %u %u %u %.4f %.2f %.1f %.2f\n",which, numNeurons, locust.sumPN, locust.sumKC, locust.sumLHI, locust.sumDN, timer.getElapsedTime(),VDN[0], TOTAL_TME, DT);
    fprintf(stdout, "GPU=%d, %u neurons, %u PN spikes, %u KC spikes, %u LHI spikes, %u DN spikes, simulation took %.4f secs, VDN[0]=%.2f DT=%.1f %.2f\n",which, numNeurons, locust.sumPN, locust.sumKC, locust.sumLHI, locust.sumDN, timer.getElapsedTime(),VDN[0], TOTAL_TME, DT);

    fclose(osf);
    fclose(osf2);
    fclose(timef);

#ifdef TIMING
    fclose(timeros);
#endif

#ifndef CPU_ONLY
    if (which == GPU) {
        locust.free_device_mem();
    }
#endif

    return 0;
}
