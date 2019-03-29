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

#include "../include/spike_recorder.h"

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

    /*string OutDir = string(argv[1]) + "_output";
    string name;
    unsigned int retval = 0; //make the compiler happy

    name= OutDir+ "/" + argv[1]+ ".time";
    FILE *timef= fopen(name.c_str(),"a");


    name= OutDir+ "/"+ argv[1] + ".out.Vm";
    cout << name << endl;
    FILE *osf= fopen(name.c_str(),"w");
    if (which == CPU) {
        name= OutDir+ "/" + argv[1] + ".out.St.CPU";
    }
    else {
        name= OutDir+ "/" + argv[1] + ".out.St.GPU";
    }
    FILE *osf2= fopen(name.c_str(),"w");

    //-----------------------------------------------------------------
    // build the neuronal circuitry
    fprintf(stdout, "#creating classIzh\n");
    classIzh PCNN;
    PCNN.initializeAllVars(which);
    fprintf(stdout, "#classIzh created\n");

    //open log file
    string logname=OutDir+ "/logfile";
    //gettimeofday(&timeforlog, NULL);
    FILE *flog= fopen(logname.c_str(),"a");

    time_t timeforlog=time(0);
    tm * timeinfo = localtime(&timeforlog);
    fprintf(flog,"%d/%d/%d, %d:%d\n",timeinfo->tm_mday,timeinfo->tm_mon+1,timeinfo->tm_year+1900,timeinfo->tm_hour,timeinfo->tm_min);

    fprintf(flog,"Izh_sparse_sim, ");
    fprintf(flog, "# DT %f \n", DT);
    fprintf(flog, "# T_REPORT_TME %f \n", T_REPORT_TME);
    fprintf(flog, "# TOTAL_TME %f \n", TOTAL_TME);

    unsigned int sumSynapses=0;

    fprintf(stdout, "#reading synapses ... \n");
    FILE *f_info, *f, *f_indInG,*f_ind;
    unsigned int connN;*/

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

    fprintf(stdout, "# neuronal circuitry built, start computation ... \n\n");

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation


    fprintf(stdout, "# DT %f \n", DT);
    fprintf(stdout, "# T_REPORT_TME %f \n", REPORT_TIME);
    fprintf(stdout, "# TOTAL_TME %f \n", TOTAL_TIME);

    fprintf(stdout, "# We are running with fixed time step %f \n", DT);
    fprintf(stdout, "# initial wait time execution ... \n");

    SpikeRecorder excSpikes(outLabel + "_exc_st", glbSpkCntPExc, glbSpkPExc);
    SpikeRecorder inhSpikes(outLabel + "_inh_st", glbSpkCntPInh, glbSpkPInh);

    while(t < TOTAL_TIME) {
        stepTime();

        pullPExcCurrentSpikesFromDevice();
        pullPInhCurrentSpikesFromDevice();

        excSpikes.record(t);
        inhSpikes.record(t);

    }

    /*timer.stopTimer();

    cout << "Output files are created under the current directory. Output and parameters are logged in: " << logname << endl;
    fprintf(timef, "%d %d %u %u %.4f %.2f %.1f %.2f %u %s %d\n",which, _NExc + _NInh, PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, logname.c_str(), _FTYPE);
    fprintf(flog, "%u neurons in total\n%u spikes in the excitatory population\n%u spikes in the inhibitory population\nElapsed time is %.4f\nLast Vm value of the 1st neuron is %.2f\nTotal time %f at DT+%f \nTotal number of synapses in the model is %u, %d precision\n\n#################\n", _NExc + _NInh, PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, _FTYPE);
    fprintf(stdout, "%u neurons in total\n%u spikes in the excitatory population\n%u spikes in the inhibitory population\nElapsed time is %.4f\nLast Vm value of the 1st neuron is %.2f\nTotal simulation of %f ms at DT=%f \nTotal number of synapses in the model is %u, %d precision\n\n#################\n", _NExc + _NInh, PCNN.sumPExc, PCNN.sumPInh, timer.getElapsedTime(),VPExc[0], TOTAL_TME, DT, sumSynapses, _FTYPE);
    fclose(osf);
    fclose(timef);
    fclose(osf2);
    fclose(flog);*/

    return EXIT_SUCCESS;
}


