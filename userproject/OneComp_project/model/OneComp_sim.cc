/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/
#include "OneComp_model.h"

#include "hr_time.h"
#include "modelSpec.h"

#include "sizes.h"

//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 100.0
#define TOTAL_TME 5000

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: OneComp_sim <basename> <CPU=0, GPU=1> \n");
        return 1;
    }
    int which= atoi(argv[2]);
    string OutDir = string(argv[1]) +"_output";
    string name, name2;

    name= OutDir+ "/" + argv[1]+ ".time";
    FILE *timef= fopen(name.c_str(),"a");

    CStopWatch timer;
    timer.startTimer();
    fprintf(stderr, "# DT %f \n", DT);
    fprintf(stderr, "# T_REPORT_TME %f \n", T_REPORT_TME);
    fprintf(stderr, "# TOTAL_TME %d \n", TOTAL_TME);

    name= OutDir+ "/" + argv[1] + ".out.Vm";
    FILE *osf= fopen(name.c_str(),"w");
    name2= OutDir+ "/" + argv[1] + ".explinp";
    FILE *osf2= fopen(name2.c_str(),"w");
    //-----------------------------------------------------------------
    // build the neuronal circuitry
    neuronpop IzhikevichPop;

    IzhikevichPop.init(which);         // this includes copying g's for the GPU version

    fprintf(stderr, "# neuronal circuitry built, start computation ... \n\n");
    const unsigned int outno = (_NC1 > 10) ? 10 : _NC1;

    //------------------------------------------------------------------
    // output general parameters to output file and start the simulation

    fprintf(stderr, "# We are running with fixed time step %f \n", DT);
    fprintf(stderr, "# initial wait time execution ... \n");

    t= 0.0;
    int done= 0;
    float last_t_report=  t;
    while (!done)
    {
        IzhikevichPop.run(DT, which); // run next batch
#ifndef CPU_ONLY
        if (which == GPU) {
            IzhikevichPop.getSpikeNumbersFromGPU();
            CHECK_CUDA_ERRORS(cudaMemcpy(VIzh1, d_VIzh1, outno*sizeof(scalar), cudaMemcpyDeviceToHost));
        }
#endif
        IzhikevichPop.sum_spikes();
        fprintf(osf, "%f ", t);

        for(int i=0;i<outno;i++) {
            fprintf(osf, "%f ", VIzh1[i]);
        }
        fprintf(osf, "\n");

        // report progress
        if (t - last_t_report >= T_REPORT_TME)
        {
            fprintf(stderr, "time %f \n", t);
            last_t_report= t;
        }

        done= (t >= TOTAL_TME);
    }

    timer.stopTimer();
    fprintf(timef, "%d %d %u %f %f \n",which, _NC1, IzhikevichPop.sumIzh1, timer.getElapsedTime(),VIzh1[0]);
    //  cerr << "Output files are created under the current directory." << endl;
    cout << timer.getElapsedTime() << endl;
    fclose(osf);
    fclose(osf2);
    fclose(timef);

    return 0;
}
